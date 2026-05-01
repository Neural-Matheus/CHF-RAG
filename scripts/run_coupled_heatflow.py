import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_questions, load_legal_docs
from src.embedder import Embedder, EMBEDDER_CONFIGS
from src.baselines import evaluate_cosine, evaluate_cosine_doc_level, ir_metrics
from src.ccg_rag import build_cocitation_graph
from src.coupled_heatflow import (
    classify_doc_hierarchy,
    build_semantic_laplacian,
    build_cocitation_laplacian,
    build_hierarchy_laplacian,
    build_coupled_operator,
    initial_distribution,
    retrieve_at_time,
    search_optimal_t,
)
from src.heatflow_metrics import (
    erc_star,
    auc_f1,
    aggregate_t_star,
    energy_along_trajectory,
    empirical_decay_rate,
    cheeger_bound_check,
    spectral_bound_check,
    flow_signatures_for_topk,
    provenance_entropy,
    coupling_lift,
    coupling_work_ratio,
    operator_diagnostics,
)
from src.question_typing import (
    classify_question_type,
    classify_question_by_refs,
    classify_question_by_ref_count,
    count_diploma_mentions,
    stratify_t_star,
)


OUTPUT_DIR = Path("/app/outputs/coupled_heatflow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE  = Path("/app/outputs/benchmark/embeddings")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "chf_run.log"),
    ],
)
logger = logging.getLogger("chf_rag")

TOP_K            = 10
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
MAX_QUESTIONS    = 200
K_NEIGHBORS_SEM  = 8


T_GRID = np.concatenate([
    np.array([0.0]),
    np.logspace(-2, 1.5, 30),    
])

DEFAULT_BETA      = 0.10
GAMMA_ERC         = 0.5     

BETA_SWEEP        = [0.05, 0.10, 0.30, 0.50, 1.00]

WEIGHT_SWEEP = {
    "w_balanced":     (1/3, 1/3, 1/3),
    "w_sem_heavy":    (0.6, 0.2, 0.2),
    "w_cit_heavy":    (0.2, 0.6, 0.2),
    "w_hier_heavy":   (0.2, 0.2, 0.6),
    "w_struct_heavy": (0.2, 0.4, 0.4),  
}

EMBEDDERS = [
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "rufimelo/Legal-BERTimbau-sts-large-ma-v3",
    "stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0",
]

STRUCT_ABLATIONS = {
    "sem_only":   dict(beta_sc=0, beta_sh=0, beta_cs=0, beta_ch=0, beta_hs=0, beta_hc=0,
                       use_cit=False, use_hier=False),
    "cit_only":   dict(beta_sc=0, beta_sh=0, beta_cs=0, beta_ch=0, beta_hs=0, beta_hc=0,
                       use_sem=False, use_hier=False),
    "hier_only":  dict(beta_sc=0, beta_sh=0, beta_cs=0, beta_ch=0, beta_hs=0, beta_hc=0,
                       use_sem=False, use_cit=False),
    "sem_cit":    dict(beta_sc=DEFAULT_BETA, beta_sh=0, beta_cs=DEFAULT_BETA, beta_ch=0,
                       beta_hs=0, beta_hc=0, use_hier=False),
}

def _chf_full_cfg(beta: float) -> Dict:
    return dict(
        beta_sc=beta, beta_sh=beta,
        beta_cs=beta, beta_ch=beta,
        beta_hs=beta, beta_hc=beta,
    )


def cache_path(model_name: str, kind: str) -> Path:
    safe = model_name.replace("/", "__")
    return EMB_CACHE / f"{safe}_{kind}.npy"


def load_or_embed(model_name: str, questions, all_chunks):
    qp, cp = cache_path(model_name, "questions"), cache_path(model_name, "chunks")
    if qp.exists() and cp.exists():
        logger.info("  Cache de embeddings: %s", model_name.split("/")[-1])
        return np.load(qp), np.load(cp)
    logger.info("  Gerando embeddings: %s ...", model_name.split("/")[-1])
    emb = Embedder(model_name=model_name)
    q_embs = emb.embed_queries([q.text for q in questions])
    c_embs = emb.embed_passages([c.text for c in all_chunks])
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(qp, q_embs)
    np.save(cp, c_embs)
    return q_embs, c_embs


def build_operator_for_ablation(
    name: str,
    L_sem, L_cit, L_hier,
    cfg: Dict,
):

    N = L_sem.shape[0]
    I = np.eye(N)

    use_sem  = cfg.get("use_sem", True)
    use_cit  = cfg.get("use_cit", True)
    use_hier = cfg.get("use_hier", True)

    Ls = L_sem  if use_sem  else 1.0 * I
    Lc = L_cit  if use_cit  else 1.0 * I
    Lh = L_hier if use_hier else 1.0 * I

    op = build_coupled_operator(
        Ls, Lc, Lh,
        beta_sc=cfg.get("beta_sc", 0.0),
        beta_sh=cfg.get("beta_sh", 0.0),
        beta_cs=cfg.get("beta_cs", 0.0),
        beta_ch=cfg.get("beta_ch", 0.0),
        beta_hs=cfg.get("beta_hs", 0.0),
        beta_hc=cfg.get("beta_hc", 0.0),
    )
    return op, (use_sem, use_cit, use_hier)


def _fill_baseline_defaults(r: Dict) -> None:
    r.setdefault("f1_at_zero", r.get("f1_k", 0.0))
    r.setdefault("delta_f1", 0.0)
    r.setdefault("t_star", 0.0)
    r.setdefault("erc_star", 0.0)
    r.setdefault("auc_f1", 0.0)
    r.setdefault("provenance_entropy", float("nan"))
    r.setdefault("cwr_sem", float("nan"))
    r.setdefault("cwr_cit", float("nan"))
    r.setdefault("cwr_hier", float("nan"))
    r.setdefault("cwr_mean", float("nan"))
    r.setdefault("question_type", "unknown")
    r.setdefault("question_type_granular", "unknown")
    r.setdefault("n_refs_explicit", 0)
    r.setdefault("n_diploma_mentions", 0)
    r.pop("reference", None)
    r.pop("faithfulness", None)
    r.pop("xai_explanation", None)


def evaluate_ablation(
    name: str,
    operator,
    L_sem, L_cit, L_hier,
    used_channels: Tuple[bool, bool, bool],
    questions: List,
    q_embs, c_embs,
    docs: List[str],
    all_chunks,
    embedder_label: str,
    k: int = TOP_K,
    ts: np.ndarray = T_GRID,
    save_signatures: bool = False,
    n_signature_samples: int = 5,
    weights_override: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[Dict], np.ndarray, List[Dict]]:

    use_sem, use_cit, use_hier = used_channels
    if weights_override is not None:
        w_s, w_c, w_h = weights_override
        s_total = w_s + w_c + w_h
        if s_total > 0:
            w_s, w_c, w_h = w_s / s_total, w_c / s_total, w_h / s_total
    else:

        weights = np.array([float(use_sem), float(use_cit), float(use_hier)])
        if weights.sum() == 0:
            weights = np.array([1, 1, 1])
        weights = weights / weights.sum()
        w_s, w_c, w_h = weights

    rows = []
    f1_curves = np.zeros((len(questions), len(ts)))
    signatures: List[Dict] = []

    for qi, question in enumerate(questions):
        if not question.all_ref_files:
            continue
        q_emb = q_embs[qi]
        U0 = initial_distribution(q_emb, c_embs, all_chunks, docs, temperature=10.0)

        t_star, f1_curve, _ = search_optimal_t(
            operator, U0, question, docs, all_chunks, c_embs, q_emb,
            ts=ts, k=k, w_s=w_s, w_c=w_c, w_h=w_h,
        )
        f1_curves[qi] = f1_curve

        retrieved, _ = retrieve_at_time(
            operator, U0, t_star, docs, all_chunks, c_embs, q_emb,
            k=k, w_s=w_s, w_c=w_c, w_h=w_h,
        )
        m = ir_metrics(retrieved, question)

        f1_zero = float(f1_curve[0])
        delta_f1 = m["f1_k"] - f1_zero
        erc = erc_star(f1_curve, ts, gamma=GAMMA_ERC)
        auc = auc_f1(f1_curve, ts)

        H_prov = float("nan")
        cwr_sem = float("nan")
        cwr_cit = float("nan")
        cwr_hier = float("nan")
        cwr_mean = float("nan")
        if int(use_sem) + int(use_cit) + int(use_hier) >= 2:
            sigs = flow_signatures_for_topk(
                operator, U0, L_sem, L_cit, L_hier, docs, retrieved,
                t_star=t_star, n_steps=20,
            )
            H_prov = provenance_entropy(sigs)


            has_coupling = any(b > 1e-9 for b in operator.betas)
            if has_coupling and t_star > 0:
                cwr = coupling_work_ratio(
                    operator, U0, L_sem, L_cit, L_hier,
                    t_star=t_star, n_steps=20,
                )
                cwr_sem  = cwr["cwr_sem"]
                cwr_cit  = cwr["cwr_cit"]
                cwr_hier = cwr["cwr_hier"]
                cwr_mean = cwr["cwr_mean"]

            if save_signatures and qi < n_signature_samples:
                for s in sigs:
                    signatures.append({
                        "question_number": question.number,
                        "strategy": name,
                        "doc_filename": s.doc_filename,
                        "chunk_id": s.chunk_id,
                        "flow_sem": s.flow_sem,
                        "flow_cit": s.flow_cit,
                        "flow_hier": s.flow_hier,
                        "rel_sem": s.rel_sem,
                        "rel_cit": s.rel_cit,
                        "rel_hier": s.rel_hier,
                    })

        rows.append({
            "question_number": question.number,
            "strategy": name,
            "embedder": embedder_label,
            "question_type": classify_question_by_refs(question.references_explicit, question.references_implicit),
            "question_type_granular": classify_question_by_ref_count(question.references_explicit, question.references_implicit),
            "n_refs_explicit": len(question.references_explicit) if question.references_explicit else 0,
            "n_diploma_mentions": count_diploma_mentions(question.text),
            "f1_k": m["f1_k"],
            "precision_k": m["precision_k"],
            "recall_k": m["recall_k"],
            "f1_at_zero": f1_zero,
            "delta_f1": delta_f1,
            "t_star": t_star,
            "erc_star": erc,
            "auc_f1": auc,
            "provenance_entropy": H_prov,
            "cwr_sem": cwr_sem,
            "cwr_cit": cwr_cit,
            "cwr_hier": cwr_hier,
            "cwr_mean": cwr_mean,
        })

    return rows, f1_curves, signatures

def main():
    logger.info("=" * 70)
    logger.info("CHF-RAG: Coupled Heat-Flow Retrieval-Augmented Generation")
    logger.info("=" * 70)
    logger.info("Limite: %d perguntas | top-k=%d", MAX_QUESTIONS, TOP_K)
    logger.info("T grid: %d pontos de %.4f a %.2f",
                len(T_GRID), T_GRID[0], T_GRID[-1])
    logger.info("β default = %.3f | γ_ERC = %.2f", DEFAULT_BETA, GAMMA_ERC)


    logger.info("\nCarregando dataset ...")
    questions = load_questions()
    _, chunks_by_doc = load_legal_docs(chunk_size=CHUNK_SIZE,
                                       chunk_overlap=CHUNK_OVERLAP)
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    docs = sorted(chunks_by_doc.keys())
    logger.info("Perguntas: %d | Chunks: %d | Documentos: %d",
                len(questions), len(all_chunks), len(docs))


    eval_questions = [q for q in questions[:MAX_QUESTIONS] if q.all_ref_files]
    eval_idx = [i for i, q in enumerate(questions[:MAX_QUESTIONS])
                if q.all_ref_files]
    logger.info("Perguntas avaliadas (com GT): %d", len(eval_questions))


    logger.info("\nClassificando hierarquia normativa ...")
    doc_levels = {d: classify_doc_hierarchy(d)[1] for d in docs}
    doc_categories = {d: classify_doc_hierarchy(d)[0] for d in docs}
    from collections import Counter
    cat_counts = Counter(doc_categories.values())
    logger.info("Distribuição por categoria: %s", dict(cat_counts.most_common()))


    outros = [d for d, c in doc_categories.items() if c == "outros"]
    if outros:
        logger.info("Top-30 filenames em 'outros' (não classificados):")
        for fn in outros[:30]:
            logger.info("  %s", fn)
        logger.info("(Total em 'outros': %d/%d)", len(outros), len(docs))


    from src.coupled_heatflow import diagnose_hierarchy_classification
    hier_diag = diagnose_hierarchy_classification(docs, show_unclassified_top=30)
    logger.info("Cobertura da classificação: %.1f%% (%d/%d docs classificados)",
                100 * hier_diag["coverage_rate"],
                hier_diag["classified"], hier_diag["total"])
    logger.info("Detalhe por categoria: %s", hier_diag["counts_by_category"])
    if hier_diag["unclassified_sample"]:
        logger.info("Amostra de até 30 filenames NÃO classificados (caem em 'outros'):")
        for fn in hier_diag["unclassified_sample"]:
            logger.info("  - %s", fn)

    with open(OUTPUT_DIR / "hierarchy_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump({
            "coverage_rate": hier_diag["coverage_rate"],
            "counts_by_category": hier_diag["counts_by_category"],
            "all_unclassified": [fn for fn in docs
                                  if classify_doc_hierarchy(fn)[1] == 0],
        }, f, ensure_ascii=False, indent=2)


    logger.info("\nGrafo de co-citação ...")
    graph = build_cocitation_graph(questions, min_cooccurrences=1)


    all_rows: List[Dict] = []
    all_signatures: List[Dict] = []
    all_operator_info: Dict[str, Dict] = {}
    f1_curves_full: Dict[str, np.ndarray] = {}

    for emb_idx, model_name in enumerate(EMBEDDERS):
        cfg = EMBEDDER_CONFIGS.get(model_name, {})
        label = cfg.get("label", model_name.split("/")[-1])

        logger.info("\n" + "═" * 60)
        logger.info("EMBEDDER %d/%d: %s", emb_idx + 1, len(EMBEDDERS), label)
        logger.info("═" * 60)

        q_embs_all, c_embs = load_or_embed(model_name, questions, all_chunks)
        q_embs_eval = q_embs_all[eval_idx]


        logger.info("\nConstruindo Laplacianos ...")
        L_sem  = build_semantic_laplacian(c_embs, all_chunks, docs,
                                          k_neighbors=K_NEIGHBORS_SEM)
        L_cit  = build_cocitation_laplacian(graph, docs)
        L_hier = build_hierarchy_laplacian(docs, doc_levels)


        n_total_strats = (
            2
            + len(STRUCT_ABLATIONS)
            + len(BETA_SWEEP)
            + len(WEIGHT_SWEEP)
        )
        step = 0


        step += 1
        logger.info("\n[%d/%d] cosine (chunk-level) ...", step, n_total_strats)
        cos_rows = evaluate_cosine(eval_questions, q_embs_eval, c_embs,
                                   all_chunks, label, k=TOP_K)

        q_by_num = {str(q.number): q for q in eval_questions}
        for r in cos_rows:
            r["strategy"] = "cosine"
            q = q_by_num.get(str(r["question_number"]))
            if q is not None:
                r["question_type"] = classify_question_by_refs(q.references_explicit, q.references_implicit)
                r["question_type_granular"] = classify_question_by_ref_count(q.references_explicit, q.references_implicit)
                r["n_refs_explicit"] = len(q.references_explicit) if q.references_explicit else 0
                r["n_diploma_mentions"] = count_diploma_mentions(q.text)
            _fill_baseline_defaults(r)
        all_rows.extend(cos_rows)


        step += 1
        logger.info("\n[%d/%d] cosine_doc_level ...", step, n_total_strats)
        cos_doc_rows = evaluate_cosine_doc_level(eval_questions, q_embs_eval,
                                                  c_embs, all_chunks, label, k=TOP_K)
        for r in cos_doc_rows:
            r["strategy"] = "cosine_doc_level"
            q = q_by_num.get(str(r["question_number"]))
            if q is not None:
                r["question_type"] = classify_question_by_refs(q.references_explicit, q.references_implicit)
                r["question_type_granular"] = classify_question_by_ref_count(q.references_explicit, q.references_implicit)
                r["n_refs_explicit"] = len(q.references_explicit) if q.references_explicit else 0
                r["n_diploma_mentions"] = count_diploma_mentions(q.text)
            _fill_baseline_defaults(r)
        all_rows.extend(cos_doc_rows)


        for ab_name, ab_cfg in STRUCT_ABLATIONS.items():
            step += 1
            logger.info("\n[%d/%d] %s ...", step, n_total_strats, ab_name)
            op, used = build_operator_for_ablation(ab_name, L_sem, L_cit, L_hier, ab_cfg)
            all_operator_info[f"{label}__{ab_name}"] = {
                **operator_diagnostics(op),
                "used_channels": list(used),
                "betas": list(op.betas),
            }
            rows, f1_curves, _ = evaluate_ablation(
                ab_name, op, L_sem, L_cit, L_hier, used,
                eval_questions, q_embs_eval, c_embs, docs, all_chunks,
                label, k=TOP_K, ts=T_GRID,
                save_signatures=False,
            )
            all_rows.extend(rows)


        op_default = None
        for beta in BETA_SWEEP:
            step += 1
            ab_name = f"chf_full_b{beta:.2f}"
            logger.info("\n[%d/%d] %s (β=%.2f) ...", step, n_total_strats, ab_name, beta)
            cfg = _chf_full_cfg(beta)
            op, used = build_operator_for_ablation(ab_name, L_sem, L_cit, L_hier, cfg)
            all_operator_info[f"{label}__{ab_name}"] = {
                **operator_diagnostics(op),
                "used_channels": list(used),
                "betas": list(op.betas),
                "beta_value": beta,
            }
            rows, f1_curves, sigs = evaluate_ablation(
                ab_name, op, L_sem, L_cit, L_hier, used,
                eval_questions, q_embs_eval, c_embs, docs, all_chunks,
                label, k=TOP_K, ts=T_GRID,
                save_signatures=(abs(beta - DEFAULT_BETA) < 1e-9),
                n_signature_samples=10,
            )
            all_rows.extend(rows)
            all_signatures.extend(sigs)

            if abs(beta - DEFAULT_BETA) < 1e-9:
                f1_curves_full[label] = f1_curves
                op_default = op

                if len(eval_questions) > 0:
                    qi0 = 0
                    U0 = initial_distribution(q_embs_eval[qi0], c_embs,
                                              all_chunks, docs, temperature=10.0)
                    ts_e = np.linspace(0, T_GRID[-1], 60)
                    E = energy_along_trajectory(op, U0, ts_e)
                    decay = empirical_decay_rate(E, ts_e)
                    diag = operator_diagnostics(op)
                    bounds = spectral_bound_check(
                        E, ts_e,
                        lambda_min=diag["lambda_min"],
                        spectral_gap=op.spectral_gap,
                        operator=op, U0=U0,
                    )
                    pd.DataFrame({"t": ts_e, "energy": E}).to_csv(
                        OUTPUT_DIR / f"chf_energy_{label}_q{eval_questions[0].number}.csv",
                        index=False
                    )
                    logger.info(
                        "  Energia: E(0)=%.4e → E(T)=%.4e | λ_emp=%.4f | "
                        "λ_min=%.4e | λ_2=%.4f",
                        float(E[0]), float(E[-1]), decay,
                        diag["lambda_min"], op.spectral_gap)
                    logger.info(
                        "  Bound (raw E): strong [E≤E0·exp(-2λ_min·t)] violated=%s ratio=%.3f",
                        bounds["bound_strong_violated"], bounds["bound_strong_max_ratio"])
                    logger.info(
                        "  Bound (raw E): optimistic [E≤E0·exp(-λ_2·t)] violated=%s ratio=%.3f "
                        "(esperado violar se U₀ não ⊥ ker(M))",
                        bounds["bound_optimistic_violated"], bounds["bound_optimistic_max_ratio"])
                    if "projected_bound_violated" in bounds:
                        logger.info(
                            "  Bound (projected E): [E_proj≤E_proj0·exp(-λ_2·t)] violated=%s ratio=%.3f",
                            bounds["projected_bound_violated"], bounds["projected_max_ratio"])


        if op_default is None:
            logger.warning("Operador β=%.2f não foi construído; pulando w-sweep.",
                           DEFAULT_BETA)
        else:
            for w_name, weights_tuple in WEIGHT_SWEEP.items():
                step += 1
                ab_name = f"chf_full_{w_name}"
                logger.info("\n[%d/%d] %s (w=%s) ...", step, n_total_strats,
                            ab_name, weights_tuple)
                rows, _, _ = evaluate_ablation(
                    ab_name, op_default, L_sem, L_cit, L_hier,
                    used_channels=(True, True, True),
                    questions=eval_questions, q_embs=q_embs_eval, c_embs=c_embs,
                    docs=docs, all_chunks=all_chunks,
                    embedder_label=label, k=TOP_K, ts=T_GRID,
                    save_signatures=False,
                    weights_override=weights_tuple,
                )
                all_rows.extend(rows)


    if not all_rows:
        logger.error("Nenhum resultado gerado.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "chf_raw.csv", index=False, encoding="utf-8")


    metric_cols = ["f1_k", "precision_k", "recall_k",
                   "f1_at_zero", "delta_f1", "t_star",
                   "erc_star", "auc_f1", "provenance_entropy",
                   "cwr_sem", "cwr_cit", "cwr_hier", "cwr_mean"]
    avail = [c for c in metric_cols if c in df.columns]
    summary = (df.groupby(["strategy", "embedder"])[avail]
                 .mean(numeric_only=True).round(4).reset_index())
    counts = df.groupby(["strategy", "embedder"]).size().reset_index(name="n_q")
    summary = summary.merge(counts, on=["strategy", "embedder"])
    summary.to_csv(OUTPUT_DIR / "chf_summary.csv", index=False, encoding="utf-8")


    t_star_dist = (df[~df["strategy"].isin(["cosine", "cosine_doc_level"])]
                   .groupby(["strategy", "embedder"])["t_star"]
                   .agg(["min", "median", "mean", "max", "std"])
                   .round(4).reset_index())
    t_star_dist.to_csv(OUTPUT_DIR / "chf_t_star_dist.csv",
                       index=False, encoding="utf-8")


    if all_signatures:
        pd.DataFrame(all_signatures).to_csv(
            OUTPUT_DIR / "chf_signatures.csv", index=False, encoding="utf-8"
        )


    with open(OUTPUT_DIR / "chf_operator_info.json", "w", encoding="utf-8") as f:
        json.dump(all_operator_info, f, ensure_ascii=False, indent=2)


    for label, curves in f1_curves_full.items():
        np.save(OUTPUT_DIR / f"chf_f1_curves_{label}.npy", curves)
        np.save(OUTPUT_DIR / f"chf_t_grid.npy", T_GRID)


    logger.info("\n" + "=" * 70)
    logger.info("RESULTADO FINAL — sumário (médias por estratégia)")
    logger.info("=" * 70)
    short_cols = ["strategy", "embedder", "f1_k", "precision_k", "recall_k",
                  "t_star", "erc_star", "auc_f1", "n_q"]
    short = summary[[c for c in short_cols if c in summary.columns]]
    logger.info("\n%s", short.to_string(index=False))


    logger.info("\n" + "=" * 70)
    logger.info("ΔF1@k vs cosine_doc_level (baseline JUSTO p/ CHF)")
    logger.info("=" * 70)
    for emb in summary["embedder"].unique():
        sub = summary[summary["embedder"] == emb]
        baseline_row = sub[sub["strategy"] == "cosine_doc_level"]
        if not len(baseline_row):
            continue
        baseline_f1 = float(baseline_row.iloc[0]["f1_k"])
        baseline_recall = float(baseline_row.iloc[0]["recall_k"])
        baseline_prec = float(baseline_row.iloc[0]["precision_k"])
        logger.info("  [%s] cosine_doc_level: F1=%.4f  P=%.4f  R=%.4f",
                    emb, baseline_f1, baseline_prec, baseline_recall)
        for strat in sub["strategy"].unique():
            if strat in ("cosine", "cosine_doc_level"):
                continue
            row = sub[sub["strategy"] == strat].iloc[0]
            logger.info("  [%s] %-22s F1=%.4f Δ=%+.4f | "
                        "P=%.4f Δ=%+.4f | R=%.4f Δ=%+.4f",
                        emb, strat,
                        float(row["f1_k"]),     float(row["f1_k"])     - baseline_f1,
                        float(row["precision_k"]), float(row["precision_k"]) - baseline_prec,
                        float(row["recall_k"]),    float(row["recall_k"])    - baseline_recall)


    logger.info("\n" + "=" * 70)
    logger.info("β-SWEEP — variação de β na chf_full")
    logger.info("=" * 70)
    for emb in summary["embedder"].unique():
        sub = summary[summary["embedder"] == emb]
        beta_rows = sub[sub["strategy"].str.startswith("chf_full_b")]
        if not len(beta_rows):
            continue
        beta_rows = beta_rows.copy()
        beta_rows["beta"] = beta_rows["strategy"].str.extract(r"b([\d\.]+)").astype(float)
        beta_rows = beta_rows.sort_values("beta")
        for _, row in beta_rows.iterrows():
            logger.info("  [%s] β=%.2f | F1=%.4f  P=%.4f  R=%.4f  t*=%.3f  CWR_mean=%.3f",
                        emb, float(row["beta"]),
                        float(row["f1_k"]), float(row["precision_k"]),
                        float(row["recall_k"]), float(row["t_star"]),
                        float(row["cwr_mean"]) if not pd.isna(row["cwr_mean"]) else float("nan"))


    logger.info("\n" + "=" * 70)
    logger.info("w-SWEEP — pesos do score (operador fixo β=%.2f)" % DEFAULT_BETA)
    logger.info("=" * 70)
    for emb in summary["embedder"].unique():
        sub = summary[summary["embedder"] == emb]
        w_rows = sub[sub["strategy"].str.startswith("chf_full_w_")]
        if not len(w_rows):
            continue
        for _, row in w_rows.iterrows():
            logger.info("  [%s] %-25s | F1=%.4f  P=%.4f  R=%.4f  t*=%.3f",
                        emb, row["strategy"],
                        float(row["f1_k"]), float(row["precision_k"]),
                        float(row["recall_k"]), float(row["t_star"]))


    logger.info("\n" + "=" * 70)
    logger.info("Coupling Lift — chf_full(β=%.2f) vs canais isolados" % DEFAULT_BETA)
    logger.info("=" * 70)
    chf_default_name = f"chf_full_b{DEFAULT_BETA:.2f}"
    for emb in summary["embedder"].unique():
        sub_df = df[df["embedder"] == emb]
        f1_full = sub_df[sub_df["strategy"] == chf_default_name]["f1_k"].values
        f1_iso = {}
        for n in ["sem_only", "cit_only", "hier_only"]:
            v = sub_df[sub_df["strategy"] == n]["f1_k"].values
            if len(v) == len(f1_full):
                f1_iso[n] = v
        if f1_iso and len(f1_full) > 0:
            lifts = coupling_lift(f1_full, f1_iso)
            for k, v in lifts.items():
                logger.info("  [%s] %s = %+.4f", emb, k, v)


    logger.info("\n" + "=" * 70)
    logger.info("Coupling Work Ratio (CWR) — quanto o termo de troca de massa trabalha")
    logger.info("=" * 70)
    logger.info("  CWR ≈ 0    → canal evolui isolado (acoplamento inerte)")
    logger.info("  CWR > 0.1  → acoplamento contribui significativamente")
    for emb in summary["embedder"].unique():
        sub = summary[summary["embedder"] == emb]

        for _, row in sub[sub["strategy"].str.startswith("chf_full_b")].iterrows():
            logger.info("  [%s] %-22s | CWR sem=%.3f cit=%.3f hier=%.3f mean=%.3f",
                        emb, row["strategy"],
                        float(row["cwr_sem"]) if not pd.isna(row["cwr_sem"]) else float("nan"),
                        float(row["cwr_cit"]) if not pd.isna(row["cwr_cit"]) else float("nan"),
                        float(row["cwr_hier"]) if not pd.isna(row["cwr_hier"]) else float("nan"),
                        float(row["cwr_mean"]) if not pd.isna(row["cwr_mean"]) else float("nan"))


    logger.info("\n" + "=" * 70)
    logger.info("STRATIFIED t* — diretas vs conceituais")
    logger.info("=" * 70)
    logger.info("Hipótese: t*_conceitual > t*_direto (perguntas conceituais")
    logger.info("precisam de mais difusão para encontrar os documentos certos)")


    n_total = (df.groupby(["question_number", "embedder"])
                 .first().reset_index())
    if "question_type" in n_total.columns:
        type_counts = n_total["question_type"].value_counts().to_dict()
        logger.info("Distribuição global das perguntas: %s", type_counts)


    chf_strategies = [s for s in df["strategy"].unique()
                      if s.startswith("chf_full") or s in
                      ["sem_only", "cit_only", "hier_only", "sem_cit"]]
    stratified_rows = []
    for emb in summary["embedder"].unique():
        for strat in chf_strategies:
            sub_df = df[(df["embedder"] == emb) & (df["strategy"] == strat)]
            if not len(sub_df) or "question_type" not in sub_df.columns:
                continue
            t_stars = sub_df["t_star"].values
            f1s = sub_df["f1_k"].values
            types = sub_df["question_type"].tolist()

            stats = stratify_t_star(t_stars, f1s, types)
            stratified_rows.append({
                "embedder": emb,
                "strategy": strat,
                "n_direct": stats.n_direct,
                "n_conceptual": stats.n_conceptual,
                "t_star_direct_median": stats.t_star_direct_median,
                "t_star_conceptual_median": stats.t_star_conceptual_median,
                "delta_t_star_median": stats.delta_t_star_median,
                "t_star_direct_mean": stats.t_star_direct_mean,
                "t_star_conceptual_mean": stats.t_star_conceptual_mean,
                "f1_direct_mean": stats.f1_direct_mean,
                "f1_conceptual_mean": stats.f1_conceptual_mean,
                "welch_p_value": stats.welch_t_p_value,
            })


            sig = "***" if stats.welch_t_p_value < 0.001 else \
                  "**"  if stats.welch_t_p_value < 0.01  else \
                  "*"   if stats.welch_t_p_value < 0.05  else "n.s."
            logger.info(
                "  [%s] %-22s | t*_direct=%.3f  t*_conc=%.3f  Δ=%+.3f  p=%.3e  %s",
                emb, strat,
                stats.t_star_direct_median,
                stats.t_star_conceptual_median,
                stats.delta_t_star_median,
                stats.welch_t_p_value,
                sig,
            )


    if stratified_rows:
        pd.DataFrame(stratified_rows).to_csv(
            OUTPUT_DIR / "chf_stratified_t_star.csv",
            index=False, encoding="utf-8"
        )


    logger.info("\n" + "=" * 70)
    logger.info("RECALL@k — comparação completa")
    logger.info("=" * 70)
    for emb in summary["embedder"].unique():
        sub = summary[summary["embedder"] == emb].sort_values("recall_k", ascending=False)
        for _, row in sub.iterrows():
            logger.info("  [%s] %-25s recall=%.4f  precision=%.4f  f1=%.4f",
                        emb, row["strategy"],
                        float(row["recall_k"]),
                        float(row["precision_k"]),
                        float(row["f1_k"]))

    print("\n" + "=" * 70)
    print("CHF-RAG v3 concluído")
    print("=" * 70)
    print(short.to_string(index=False))
    print(f"\nArquivos em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
