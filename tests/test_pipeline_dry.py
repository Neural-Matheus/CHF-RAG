import numpy as np

import src.baselines as baselines
import src.ccg_rag as ccg_rag
import src.coupled_heatflow as chf
import src.heatflow_metrics as mtr
from src.data_loader import LegalChunk, Question


def _build_synthetic_world(n_docs=20, chunks_per_doc=4, d_emb=24, n_queries=15, seed=42):
    rng = np.random.default_rng(seed)

    hierarchy_names = [
        "Constituicao_Federal_1988.txt",
        "Lei_Complementar_116_2003.txt",
        "Lei_Complementar_123_2006.txt",
        "Lei_7713_1988.txt",
        "Lei_9250_1995.txt",
        "Lei_8541_1992.txt",
        "Lei_11482_2007.txt",
        "Decreto_9580_2018.txt",
        "Decreto_3000_1999.txt",
        "Decreto_Lei_5844_1943.txt",
        "Instrucao_Normativa_RFB_1500_2014.txt",
        "Instrucao_Normativa_RFB_2110_2022.txt",
        "Portaria_RFB_27_2024.txt",
        "Resolucao_15_RFB.txt",
        "Ato_Declaratorio_5.txt",
    ]
    while len(hierarchy_names) < n_docs:
        hierarchy_names.append(f"outro_{len(hierarchy_names)}.txt")
    docs = hierarchy_names[:n_docs]

    n_clusters = 5
    centers = rng.normal(size=(n_clusters, d_emb))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    chunks, embs = [], []
    doc_to_cluster = {d: rng.integers(n_clusters) for d in docs}
    for d in docs:
        c = doc_to_cluster[d]
        for cid in range(chunks_per_doc):
            e = centers[c] + 0.15 * rng.normal(size=d_emb)
            e = e / np.linalg.norm(e)
            embs.append(e)
            chunks.append(LegalChunk(d, cid, f"{d}/{cid}", 0, 10))
    embs = np.stack(embs).astype(np.float32)

    q_embs, qs = [], []
    for i in range(n_queries):
        c = rng.integers(n_clusters)
        e = centers[c] + 0.1 * rng.normal(size=d_emb)
        e = e / np.linalg.norm(e)
        q_embs.append(e)
        cluster_docs = [d for d in docs if doc_to_cluster[d] == c]
        gt = list(rng.choice(cluster_docs, size=min(2, len(cluster_docs)), replace=False))
        qs.append(Question(str(i + 1), "x", "test", "a", [], [], gt, []))
    q_embs = np.stack(q_embs).astype(np.float32)

    return docs, chunks, embs, qs, q_embs


def test_full_pipeline_runs():

    docs, all_chunks, c_embs, qs, q_embs = _build_synthetic_world()
    eval_questions = [q for q in qs if q.all_ref_files]
    assert len(eval_questions) > 0


    doc_levels = {d: chf.classify_doc_hierarchy(d)[1] for d in docs}
    assert all(0 <= lv <= 5 for lv in doc_levels.values())


    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    assert graph.n_documents > 0


    L_sem = chf.build_semantic_laplacian(c_embs, all_chunks, docs, k_neighbors=5)
    L_cit = chf.build_cocitation_laplacian(graph, docs)
    L_hier = chf.build_hierarchy_laplacian(docs, doc_levels)
    for L in (L_sem, L_cit, L_hier):
        assert L.shape == (len(docs), len(docs))


    cos_doc_rows = baselines.evaluate_cosine_doc_level(
        eval_questions, q_embs, c_embs, all_chunks, "synthetic", k=5
    )
    assert len(cos_doc_rows) == len(eval_questions)
    for row in cos_doc_rows:
        assert 0.0 <= row["f1_k"] <= 1.0
        assert 0.0 <= row["recall_k"] <= 1.0


    op = chf.build_coupled_operator(
        L_sem, L_cit, L_hier,
        beta_sc=0.1, beta_sh=0.1,
        beta_cs=0.1, beta_ch=0.1,
        beta_hs=0.1, beta_hc=0.1,
        tikhonov=1e-3,
    )
    diag = mtr.operator_diagnostics(op)
    assert diag["is_psd"], f"Operator is not PSD: lambda_min={diag['lambda_min']}"
    assert diag["lambda_min"] >= 1e-3 - 1e-6


    ts = np.concatenate([np.array([0.0]), np.logspace(-2, 1.5, 10)])
    t_stars = []
    for i, q in enumerate(eval_questions):
        u0 = chf.initial_distribution(
            q_embs[i], c_embs, all_chunks, docs, temperature=10.0
        )
        t_star, _, _ = chf.search_optimal_t(
            op, u0, q, docs, all_chunks, c_embs, q_embs[i], ts,
            k=5, w_s=1/3, w_c=1/3, w_h=1/3,
        )
        assert t_star in ts, f"t*={t_star} not in grid"
        t_stars.append(t_star)
    assert len(t_stars) == len(eval_questions)
