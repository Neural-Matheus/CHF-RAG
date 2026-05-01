import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_questions, load_legal_docs
from src.embedder import Embedder, EMBEDDER_CONFIGS
from src.baselines import evaluate_cosine_doc_level, ir_metrics, ir_metrics_at_ks
from src.ccg_rag import build_cocitation_graph
from src.coupled_heatflow import (
    classify_doc_hierarchy,
    build_semantic_laplacian,
    build_cocitation_laplacian,
    build_hierarchy_laplacian,
    build_coupled_operator,
    initial_distribution,
    search_optimal_t,
)
from src.hybrid_rerank import (
    BM25DocLevel,
    reciprocal_rank_fusion,
    CrossEncoderReranker,
    evaluate_bm25_pure,
    evaluate_with_rerank,
)


OUTPUT_DIR = Path("/app/outputs/hybrid_rerank")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE = Path("/app/outputs/benchmark/embeddings")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "hybrid_rerank.log"),
    ],
)
logger = logging.getLogger("hybrid_rerank")




TOP_K          = 10
K_LIST         = [5, 10, 20]
N_CANDIDATES   = 100
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
MAX_QUESTIONS  = 200
K_NEIGHBORS_SEM = 8


CHF_BETA  = 0.10
CHF_W_S, CHF_W_C, CHF_W_H = 0.2, 0.4, 0.4


EMBEDDERS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]




RERANKERS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",

]

T_GRID = np.concatenate([
    np.array([0.0]),
    np.logspace(-2, 1.5, 30),
])




def cache_path(model_name: str, kind: str) -> Path:
    safe = model_name.replace("/", "__")
    return EMB_CACHE / f"{safe}_{kind}.npy"


def load_or_embed(model_name: str, questions, all_chunks):
    qp, cp = cache_path(model_name, "questions"), cache_path(model_name, "chunks")
    if qp.exists() and cp.exists():
        logger.info("  Cache: %s", model_name.split("/")[-1])
        return np.load(qp), np.load(cp)
    logger.info("  Gerando embeddings: %s ...", model_name.split("/")[-1])
    emb = Embedder(model_name=model_name)
    q_embs = emb.embed_queries([q.text for q in questions])
    c_embs = emb.embed_passages([c.text for c in all_chunks])
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(qp, q_embs)
    np.save(cp, c_embs)
    return q_embs, c_embs


def cosine_top_n_chunks(q_emb, c_embs, all_chunks, n=N_CANDIDATES):
    scores = c_embs @ q_emb

    doc_best = {}
    for i, c in enumerate(all_chunks):
        s = float(scores[i])
        if c.doc_filename not in doc_best or s > doc_best[c.doc_filename][1]:
            doc_best[c.doc_filename] = (i, s)

    ranked = sorted(doc_best.items(), key=lambda x: x[1][1], reverse=True)
    return [all_chunks[ci] for _, (ci, _) in ranked[:n]]


def cosine_top_n_chunks_raw(q_emb, c_embs, all_chunks, n=N_CANDIDATES):
    scores = c_embs @ q_emb
    idx = np.argsort(scores)[::-1][:n]
    return [all_chunks[i] for i in idx]


def chf_top_n_chunks(operator, U0, t_star, docs, all_chunks, c_embs, q_emb,
                     n=N_CANDIDATES, w_s=1/3, w_c=1/3, w_h=1/3):
    Ut = operator.evolve(U0, t_star)
    N = operator.N
    s = w_s * Ut[:N] + w_c * Ut[N:2*N] + w_h * Ut[2*N:]
    doc_order = np.argsort(s)[::-1]

    cos_scores = c_embs @ q_emb
    doc_to_chunks = {}
    for i, c in enumerate(all_chunks):
        doc_to_chunks.setdefault(c.doc_filename, []).append(i)

    out = []
    seen = set()
    for di in doc_order:
        d = docs[di]
        if d in seen:
            continue
        seen.add(d)
        ci_list = doc_to_chunks.get(d, [])
        if not ci_list:
            continue
        best = max(ci_list, key=lambda j: cos_scores[j])
        out.append(all_chunks[best])
        if len(out) >= n:
            break
    return out


def main():
    logger.info("=" * 78)
    logger.info("CHF-RAG vs BM25 vs Hybrid+Rerank — comparação justa")
    logger.info("=" * 78)
    logger.info("top_k=%d  n_candidates=%d  embedders=%d  rerankers=%d",
                TOP_K, N_CANDIDATES, len(EMBEDDERS), len(RERANKERS))


    logger.info("\nCarregando dataset ...")
    questions = load_questions()
    _, chunks_by_doc = load_legal_docs(chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVERLAP)
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    docs = sorted(chunks_by_doc.keys())

    eval_questions = [q for q in questions[:MAX_QUESTIONS] if q.all_ref_files]
    eval_idx = [i for i, q in enumerate(questions[:MAX_QUESTIONS]) if q.all_ref_files]
    logger.info("Perguntas avaliadas: %d  Chunks: %d  Docs: %d",
                len(eval_questions), len(all_chunks), len(docs))


    logger.info("\nIndexando BM25 ...")
    bm25 = BM25DocLevel(all_chunks)


    logger.info("\nGrafo de co-citação ...")
    graph = build_cocitation_graph(questions, min_cooccurrences=1)
    doc_levels = {d: classify_doc_hierarchy(d)[1] for d in docs}


    logger.info("\n=== Pipeline 1: BM25 puro ===")
    bm25_rows = evaluate_bm25_pure(bm25, eval_questions, "bm25", k=TOP_K, ks=K_LIST)
    f1_bm25 = np.mean([r["f1_k"] for r in bm25_rows])
    r_bm25 = np.mean([r["recall_k"] for r in bm25_rows])
    r_bm25_5 = np.mean([r["recall_5"] for r in bm25_rows])
    r_bm25_20 = np.mean([r["recall_20"] for r in bm25_rows])
    logger.info("BM25 puro: F1@10=%.4f | R@5=%.4f R@10=%.4f R@20=%.4f",
                f1_bm25, r_bm25_5, r_bm25, r_bm25_20)

    all_rows = list(bm25_rows)


    for emb_idx, model_name in enumerate(EMBEDDERS):
        cfg = EMBEDDER_CONFIGS.get(model_name, {})
        label = cfg.get("label", model_name.split("/")[-1])

        logger.info("\n" + "═" * 70)
        logger.info("EMBEDDER %d/%d: %s", emb_idx + 1, len(EMBEDDERS), label)
        logger.info("═" * 70)

        q_embs_all, c_embs = load_or_embed(model_name, questions, all_chunks)
        q_embs_eval = q_embs_all[eval_idx]


        logger.info("\n[%s] Pipeline 2: Cosine doc-level puro ...", label)
        cos_rows = evaluate_cosine_doc_level(eval_questions, q_embs_eval,
                                              c_embs, all_chunks, label,
                                              k=TOP_K, ks=K_LIST)
        for r in cos_rows:
            r["strategy"] = "cosine_doc_level"
            r.pop("reference", None)
            r.pop("faithfulness", None)
            r.pop("xai_explanation", None)
        all_rows.extend(cos_rows)
        f1_cos = np.mean([r["f1_k"] for r in cos_rows])
        r_cos_5 = np.mean([r["recall_5"] for r in cos_rows])
        r_cos = np.mean([r["recall_k"] for r in cos_rows])
        r_cos_20 = np.mean([r["recall_20"] for r in cos_rows])
        logger.info("  F1@10=%.4f | R@5=%.4f R@10=%.4f R@20=%.4f",
                    f1_cos, r_cos_5, r_cos, r_cos_20)


        logger.info("\n[%s] Pipeline 3: CHF puro (struct_heavy) ...", label)
        L_sem = build_semantic_laplacian(c_embs, all_chunks, docs,
                                          k_neighbors=K_NEIGHBORS_SEM)
        L_cit = build_cocitation_laplacian(graph, docs)
        L_hier = build_hierarchy_laplacian(docs, doc_levels)
        op = build_coupled_operator(
            L_sem, L_cit, L_hier,
            beta_sc=CHF_BETA, beta_sh=CHF_BETA, beta_cs=CHF_BETA,
            beta_ch=CHF_BETA, beta_hs=CHF_BETA, beta_hc=CHF_BETA,
        )


        K_MAX = max(K_LIST)
        chf_rows = []
        chf_t_stars = []
        for qi, q in enumerate(eval_questions):
            U0 = initial_distribution(q_embs_eval[qi], c_embs, all_chunks,
                                       docs, temperature=10.0)
            t_star, f1_curve, _ = search_optimal_t(
                op, U0, q, docs, all_chunks, c_embs, q_embs_eval[qi],
                ts=T_GRID, k=TOP_K,
                w_s=CHF_W_S, w_c=CHF_W_C, w_h=CHF_W_H,
            )
            chf_t_stars.append(t_star)

            chunks_top = chf_top_n_chunks(
                op, U0, t_star, docs, all_chunks, c_embs, q_embs_eval[qi],
                n=K_MAX, w_s=CHF_W_S, w_c=CHF_W_C, w_h=CHF_W_H,
            )
            row = {
                "question_number": q.number,
                "strategy": "chf_struct_heavy",
                "embedder": label,
                "t_star": t_star,
                **ir_metrics(chunks_top[:TOP_K], q),
                **ir_metrics_at_ks(chunks_top, q, ks=K_LIST),
            }
            chf_rows.append(row)
        all_rows.extend(chf_rows)
        f1_chf = np.mean([r["f1_k"] for r in chf_rows])
        r_chf_5 = np.mean([r["recall_5"] for r in chf_rows])
        r_chf = np.mean([r["recall_k"] for r in chf_rows])
        r_chf_20 = np.mean([r["recall_20"] for r in chf_rows])
        logger.info("  F1@10=%.4f | R@5=%.4f R@10=%.4f R@20=%.4f  (t*_med=%.3f)",
                    f1_chf, r_chf_5, r_chf, r_chf_20, np.median(chf_t_stars))


        for reranker_name in RERANKERS:
            r_label = reranker_name.split("/")[-1]
            logger.info("\n[%s × %s] Carregando reranker ...", label, r_label)
            reranker = CrossEncoderReranker(model_name=reranker_name)


            logger.info("\n[%s × %s] Pipeline 4: cosine→rerank ...", label, r_label)
            def cos_cands(q, n, q_idx={"i": 0}):
                idx = eval_questions.index(q)
                return cosine_top_n_chunks(q_embs_eval[idx], c_embs, all_chunks, n=n)
            cosre_rows = evaluate_with_rerank(
                cos_cands, reranker, eval_questions,
                strategy_name=f"cosine→{r_label}",
                embedder_label=label,
                n_candidates=N_CANDIDATES, k=TOP_K, ks=K_LIST,
            )
            all_rows.extend(cosre_rows)
            f1_cr = np.mean([r["f1_k"] for r in cosre_rows])
            r_cr = np.mean([r["recall_k"] for r in cosre_rows])
            logger.info("  F1@%d = %.4f  (Δ vs cosine: %+.4f)  R@%d = %.4f",
                        TOP_K, f1_cr, f1_cr - f1_cos, TOP_K, r_cr)


            logger.info("\n[%s × %s] Pipeline 5: bm25→rerank ...", label, r_label)
            def bm25_cands(q, n):
                return [c for c, _ in bm25.retrieve(q.text, k=n)]
            bm25re_rows = evaluate_with_rerank(
                bm25_cands, reranker, eval_questions,
                strategy_name=f"bm25→{r_label}",
                embedder_label=label,
                n_candidates=N_CANDIDATES, k=TOP_K, ks=K_LIST,
            )
            all_rows.extend(bm25re_rows)
            f1_br = np.mean([r["f1_k"] for r in bm25re_rows])
            r_br = np.mean([r["recall_k"] for r in bm25re_rows])
            logger.info("  F1@%d = %.4f  (Δ vs bm25: %+.4f)  R@%d = %.4f",
                        TOP_K, f1_br, f1_br - f1_bm25, TOP_K, r_br)


            logger.info("\n[%s × %s] Pipeline 6: chf→rerank (key result) ...", label, r_label)
            def chf_cands(q, n):
                idx = eval_questions.index(q)
                U0 = initial_distribution(q_embs_eval[idx], c_embs, all_chunks,
                                           docs, temperature=10.0)
                t_star, _, _ = search_optimal_t(
                    op, U0, q, docs, all_chunks, c_embs, q_embs_eval[idx],
                    ts=T_GRID, k=TOP_K, w_s=CHF_W_S, w_c=CHF_W_C, w_h=CHF_W_H,
                )
                return chf_top_n_chunks(
                    op, U0, t_star, docs, all_chunks, c_embs, q_embs_eval[idx],
                    n=n, w_s=CHF_W_S, w_c=CHF_W_C, w_h=CHF_W_H,
                )
            chfre_rows = evaluate_with_rerank(
                chf_cands, reranker, eval_questions,
                strategy_name=f"chf→{r_label}",
                embedder_label=label,
                n_candidates=N_CANDIDATES, k=TOP_K, ks=K_LIST,
            )
            all_rows.extend(chfre_rows)
            f1_chfr = np.mean([r["f1_k"] for r in chfre_rows])
            r_chfr = np.mean([r["recall_k"] for r in chfre_rows])
            logger.info("  F1@%d = %.4f  (Δ vs chf: %+.4f, Δ vs cosine→rerank: %+.4f)",
                        TOP_K, f1_chfr, f1_chfr - f1_chf, f1_chfr - f1_cr)
            logger.info("  R@%d = %.4f  (Δ vs cosine→rerank: %+.4f)",
                        TOP_K, r_chfr, r_chfr - r_cr)


            logger.info("\n[%s × %s] Pipeline 7: hybrid_rrf→rerank ...", label, r_label)
            def hybrid_cands(q, n):
                idx = eval_questions.index(q)
                bm25_top = [c for c, _ in bm25.retrieve(q.text, k=n)]
                cos_top = cosine_top_n_chunks(q_embs_eval[idx], c_embs, all_chunks, n=n)
                fused = reciprocal_rank_fusion([bm25_top, cos_top], k=60)
                return [c for c, _ in fused[:n]]
            hyre_rows = evaluate_with_rerank(
                hybrid_cands, reranker, eval_questions,
                strategy_name=f"hybrid_rrf→{r_label}",
                embedder_label=label,
                n_candidates=N_CANDIDATES, k=TOP_K, ks=K_LIST,
            )
            all_rows.extend(hyre_rows)
            f1_hr = np.mean([r["f1_k"] for r in hyre_rows])
            logger.info("  F1@%d = %.4f", TOP_K, f1_hr)


            logger.info("\n[%s × %s] Pipeline 8: hybrid(BM25,CHF)→rerank ...", label, r_label)
            def hybrid_chf_cands(q, n):
                bm25_top = [c for c, _ in bm25.retrieve(q.text, k=n)]
                chf_top = chf_cands(q, n)
                fused = reciprocal_rank_fusion([bm25_top, chf_top], k=60)
                return [c for c, _ in fused[:n]]
            hychfre_rows = evaluate_with_rerank(
                hybrid_chf_cands, reranker, eval_questions,
                strategy_name=f"hybrid_chf_rrf→{r_label}",
                embedder_label=label,
                n_candidates=N_CANDIDATES, k=TOP_K, ks=K_LIST,
            )
            all_rows.extend(hychfre_rows)
            f1_hcr = np.mean([r["f1_k"] for r in hychfre_rows])
            r_hcr = np.mean([r["recall_k"] for r in hychfre_rows])
            logger.info("  F1@%d = %.4f  R@%d = %.4f  (Δ vs hybrid_rrf: %+.4f)",
                        TOP_K, f1_hcr, TOP_K, r_hcr, f1_hcr - f1_hr)


            r_stats = reranker.stats()
            logger.info("\nReranker stats [%s × %s]: %s", label, r_label, r_stats)
            with open(OUTPUT_DIR / f"reranker_stats_{label}_{r_label}.json", "w") as f:
                json.dump(r_stats, f, indent=2)


    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "hybrid_raw.csv", index=False, encoding="utf-8")


    base_cols = ["f1_k", "precision_k", "recall_k"]
    multi_k_cols = []
    for kk in K_LIST:
        for prefix in ["f1", "precision", "recall"]:
            col = f"{prefix}_{kk}"
            if col in df.columns:
                multi_k_cols.append(col)
    metric_cols = base_cols + multi_k_cols

    summary = (df.groupby(["strategy", "embedder"])[metric_cols]
                 .mean(numeric_only=True).round(4).reset_index())
    counts = df.groupby(["strategy", "embedder"]).size().reset_index(name="n_q")
    summary = summary.merge(counts, on=["strategy", "embedder"])
    summary.to_csv(OUTPUT_DIR / "hybrid_summary.csv", index=False, encoding="utf-8")


    logger.info("\n" + "=" * 78)
    logger.info("RESULTADO — comparação CHF vs BM25 vs Hybrid + Rerank")
    logger.info("=" * 78)

    logger.info("\n%s", summary.to_string(index=False))


    if multi_k_cols:
        logger.info("\n" + "=" * 78)
        logger.info("RECALL@k em DIFERENTES profundidades")
        logger.info("=" * 78)
        recall_cols = [f"recall_{kk}" for kk in K_LIST if f"recall_{kk}" in df.columns]
        rec_table = summary[["strategy", "embedder"] + recall_cols].copy()
        rec_table = rec_table.sort_values(["embedder", "strategy"])
        logger.info("\n%s", rec_table.to_string(index=False))

        logger.info("\n" + "=" * 78)
        logger.info("F1@k em DIFERENTES profundidades")
        logger.info("=" * 78)
        f1_cols = [f"f1_{kk}" for kk in K_LIST if f"f1_{kk}" in df.columns]
        f1_table = summary[["strategy", "embedder"] + f1_cols].copy()
        f1_table = f1_table.sort_values(["embedder", "strategy"])
        logger.info("\n%s", f1_table.to_string(index=False))


    logger.info("\n=== RERANK LIFT (cada retriever × cada reranker) ===")
    for reranker_name in RERANKERS:
        r_label = reranker_name.split("/")[-1]
        for emb in summary["embedder"].unique():
            sub = summary[summary["embedder"] == emb]
            for retr_base, retr_re in [
                ("bm25", f"bm25→{r_label}"),
                ("cosine_doc_level", f"cosine→{r_label}"),
                ("chf_struct_heavy", f"chf→{r_label}"),
            ]:
                base = sub[sub["strategy"] == retr_base]
                ren = sub[sub["strategy"] == retr_re]
                if not len(base) or not len(ren):
                    continue
                f1_b = float(base["f1_k"].iloc[0])
                f1_r = float(ren["f1_k"].iloc[0])
                r_b = float(base["recall_k"].iloc[0])
                r_r = float(ren["recall_k"].iloc[0])
                logger.info("  [%s × %s] %s: F1 %.4f → %.4f (Δ%+.4f) | R %.4f → %.4f (Δ%+.4f)",
                            emb, r_label, retr_base, f1_b, f1_r, f1_r - f1_b,
                            r_b, r_r, r_r - r_b)

    print("\n" + "=" * 78)
    print("Hybrid+Rerank comparison concluído")
    print(f"Outputs em: {OUTPUT_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
