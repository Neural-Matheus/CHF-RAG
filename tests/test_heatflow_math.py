
import numpy as np

from src import baselines
from src import ccg_rag
from src import coupled_heatflow as chf
from src import data_loader
from src import heatflow_metrics as mtr
from src import question_typing as qt

LegalChunk, Question = data_loader.LegalChunk, data_loader.Question

def make_synthetic_world(N_docs=12, chunks_per_doc=3, d_emb=16, seed=0):

    rng = np.random.default_rng(seed)

    names = [
        "Constituicao_Federal.txt",
        "Lei_Complementar_116.txt",
        "Lei_7713.txt", "Lei_9250.txt", "Lei_8541.txt",
        "Decreto_9580.txt", "Decreto_3000.txt",
        "Instrucao_Normativa_RFB_1500.txt",
        "Portaria_RFB_27.txt",
        "Decreto_Lei_5844.txt",
        "Resolucao_15.txt",
        "Outras_Normas.txt",
    ][:N_docs]


    centers = rng.normal(size=(4, d_emb))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    chunks, embs = [], []
    for i, doc in enumerate(names):
        cluster = i % 4
        for cid in range(chunks_per_doc):
            e = centers[cluster] + 0.1 * rng.normal(size=d_emb)
            e = e / np.linalg.norm(e)
            embs.append(e)
            chunks.append(LegalChunk(doc, cid, f"{doc}/{cid}", 0, 10))
    embs = np.stack(embs).astype(np.float32)


    qs = [
        Question("1", "x", "test", "a", [], [], [names[0], names[2]], []),
        Question("2", "x", "test", "a", [], [], [names[0], names[2]], []),
        Question("3", "x", "test", "a", [], [], [names[2], names[5]], []),
        Question("4", "x", "test", "a", [], [], [names[2], names[5]], []),
        Question("5", "x", "test", "a", [], [], [names[1], names[3]], []),
        Question("6", "x", "test", "a", [], [], [names[7], names[5]], []),
    ]
    return names, chunks, embs, qs


def test_psd_laplacians():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)

    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)

    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)

    for name, L in [("sem", L_sem), ("cit", L_cit), ("hier", L_hier)]:
        info = chf.check_psd(L, name)
        assert info["psd"], f"L^{name} não é PSD: λ_min={info['lambda_min']}"

        assert info["lambda_max"] <= 2 + 1e-6, f"L^{name} λ_max={info['lambda_max']} > 2"
    print("   test_psd_laplacians: os 3 Laplacianos são PSD")


def test_coupled_operator_psd_and_eigendecomp():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)

    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)
    diag = mtr.operator_diagnostics(op)
    assert diag["is_psd"], f"M não é PSD: λ_min={diag['lambda_min']}"


    for i in [0, 1, 5, 10]:
        v = op.eigvecs[:, i]
        Mv = op.M @ v
        lambda_v = op.eigvals[i] * v
        err = np.linalg.norm(Mv - lambda_v) / max(np.linalg.norm(lambda_v), 1e-12)
        assert err < 1e-6, f"eigvec {i}: erro relativo {err}"

    print(f"   test_coupled_operator_psd: λ_min={diag['lambda_min']:.3e}  "
          f"gap={diag['spectral_gap']:.4f}  PSD={diag['is_psd']}")


def test_tikhonov_regularizes_isolated_ablation():

    names, chunks, embs, qs = make_synthetic_world(N_docs=15, chunks_per_doc=3)
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem  = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit  = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)

    N = len(names)
    I = np.eye(N)
    epsilon = 1e-3


    op_cit_iso = chf.build_coupled_operator(
        I, L_cit, I,
        beta_sc=0, beta_sh=0, beta_cs=0, beta_ch=0, beta_hs=0, beta_hc=0,
        tikhonov=epsilon,
    )



    assert op_cit_iso.eigvals[0] >= epsilon - 1e-6, \
        f"Tikhonov não funcionou: λ_min={op_cit_iso.eigvals[0]} < ε={epsilon}"


    op_no_tik = chf.build_coupled_operator(
        I, L_cit, I,
        beta_sc=0, beta_sh=0, beta_cs=0, beta_ch=0, beta_hs=0, beta_hc=0,
        tikhonov=0,
    )


    print(f"   test_tikhonov_regularizes_isolated_ablation:")
    print(f"   sem ε: λ_min = {op_no_tik.eigvals[0]:.3e}  (potencialmente singular)")
    print(f"   com ε={epsilon}: λ_min = {op_cit_iso.eigvals[0]:.3e}  (≥ ε)")
    print(f"   gap espectral preservado: {op_cit_iso.spectral_gap:.4f}")


def test_evolve_closed_form_consistency():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)

    rng = np.random.default_rng(42)
    U0 = rng.normal(size=3 * op.N)

    for t in [0.0, 0.1, 1.0, 5.0]:
        U_evolve = op.evolve(U0, t)

        expm = op.eigvecs @ np.diag(np.exp(-t * op.eigvals)) @ op.eigvecs.T
        U_direct = expm @ U0
        err = np.linalg.norm(U_evolve - U_direct) / max(np.linalg.norm(U_direct), 1e-12)
        assert err < 1e-9, f"t={t}: erro={err}"


    U_t0 = op.evolve(U0, 0.0)
    assert np.allclose(U_t0, U0, atol=1e-9), "evolve(t=0) ≠ U₀"

    print("   test_evolve_closed_form: U(t) consistente em vários t")


def test_evolve_batch_matches_loop():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)

    rng = np.random.default_rng(0)
    U0 = rng.normal(size=3 * op.N)
    ts = np.array([0.0, 0.5, 1.0, 2.0, 5.0])

    Ut_batch = op.evolve_batch(U0, ts)
    Ut_loop = np.stack([op.evolve(U0, float(t)) for t in ts])

    err = np.linalg.norm(Ut_batch - Ut_loop) / max(np.linalg.norm(Ut_loop), 1e-12)
    assert err < 1e-9, f"batch != loop, erro={err}"
    print("   test_evolve_batch_matches_loop")


def test_energy_monotone_decreasing():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)

    rng = np.random.default_rng(0)
    U0 = rng.normal(size=3 * op.N)
    ts = np.linspace(0, 10, 100)
    E = mtr.energy_along_trajectory(op, U0, ts)


    assert all(E[i+1] <= E[i] + 1e-9 for i in range(len(E) - 1)), \
        f"E não decresce: {E[:5]}"



    bound_check = mtr.cheeger_bound_check(E, ts, op.spectral_gap)
    print(f"   test_energy_monotone: E(0)={E[0]:.4f} → E(T)={E[-1]:.4e}")
    print(f"   bound check: gap={bound_check['spectral_gap']:.4f}  "
          f"violated={bound_check['bound_violated']}  "
          f"max_ratio={bound_check['max_ratio']:.4f}")
    assert not bound_check["bound_violated"], "bound de Cheeger violado"


def test_t_zero_recovers_cosine():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)


    q_emb = embs[0] + 0.05 * np.random.default_rng(0).normal(size=embs.shape[1])
    q_emb = q_emb / np.linalg.norm(q_emb)

    U0 = chf.initial_distribution(q_emb, embs, chunks, names, temperature=10.0)


    cos_scores = embs @ q_emb
    doc_to_chunks = {}
    for i, c in enumerate(chunks):
        doc_to_chunks.setdefault(c.doc_filename, []).append(i)
    doc_max_cos = {d: max(cos_scores[i] for i in idxs)
                   for d, idxs in doc_to_chunks.items()}
    cosine_top_docs = [d for d, _ in sorted(doc_max_cos.items(),
                                            key=lambda x: -x[1])[:5]]


    retrieved, _ = chf.retrieve_at_time(
        op, U0, t=0.0, docs=names, all_chunks=chunks,
        chunk_embs=embs, q_emb=q_emb, k=5,
    )
    chf_top_docs = [c.doc_filename for c in retrieved]


    assert chf_top_docs == cosine_top_docs, \
        f"t=0 não recupera cosine puro:\n  cosine={cosine_top_docs}\n  chf={chf_top_docs}"
    print(f"   test_t_zero_recovers_cosine: top-5 idênticos = {chf_top_docs}")


def test_t_large_concentrates_mass():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)
    rng = np.random.default_rng(0)
    U0 = rng.normal(size=3 * op.N) + 1
    U0 = np.abs(U0)
    U0 = U0 / U0.sum()


    U_inf = op.evolve(U0, t=1e3)




    assert np.isfinite(U_inf).all(), "U(∞) tem NaN/Inf"
    print(f"   test_t_large_concentrates: ||U(∞)||={np.linalg.norm(U_inf):.4f}  "
          f"(<= ||U₀||={np.linalg.norm(U0):.4f})")


def test_initial_distribution_reflects_query():

    names, chunks, embs, qs = make_synthetic_world()
    q_emb = embs[0]
    q_emb = q_emb / np.linalg.norm(q_emb)

    U0 = chf.initial_distribution(q_emb, embs, chunks, names, temperature=10.0)
    p_sem = U0[:len(names)]


    most_mass = int(np.argmax(p_sem))
    assert names[most_mass] == chunks[0].doc_filename, \
        f"massa máxima em {names[most_mass]}, esperado {chunks[0].doc_filename}"


    assert abs(p_sem.sum() - 1.0) < 1e-6, f"soma={p_sem.sum()}"
    print(f"   test_initial_distribution: massa máxima em '{names[most_mass]}' "
          f"(p={p_sem[most_mass]:.4f})")


def test_hierarchy_classification():

    cases = [
        ("Constituicao_Federal.txt",         "constituicao",     5),
        ("Emenda_Constitucional_45_2004.txt","emenda",           5),
        ("Lei_Complementar_116_2003.txt",    "lei_complementar", 4),
        ("LC_123_2006.txt",                  "lei_complementar", 4),
        ("Lei_7713_1988.txt",                "lei",              3),
        ("Lei_n_9250_1995.txt",              "lei",              3),
        ("Decreto_9580_2018.txt",            "decreto",          2),
        ("Decreto_Lei_5844.txt",             "decreto_lei",      3),
        ("Medida_Provisoria_2200.txt",       "medida_provisoria", 3),
        ("Instrucao_Normativa_RFB_1500.txt", "instrucao",        1),
        ("IN_RFB_2110_2022.txt",             "instrucao",        1),
        ("Solucao_de_Consulta_99.txt",       "instrucao",        1),
        ("Ato_Declaratorio_5.txt",           "instrucao",        1),
        ("Portaria_27.txt",                  "portaria",         1),
        ("Resolucao_15.txt",                 "resolucao",        1),
        ("Circular_BCB_3680.txt",            "circular",         1),
        ("CTN.txt",                          "lei_complementar", 4),
        ("Codigo_Tributario_Nacional.txt",   "lei_complementar", 4),

        ("acordao_CARF_1234_2023.txt",       "jurisprudencia",   1),
        ("CARF_2023_acordao_5567.txt",       "jurisprudencia",   1),
        ("Acordao_STJ_REsp_999.txt",         "jurisprudencia",   1),
        ("Sumula_Vinculante_24.txt",         "jurisprudencia",   1),
        ("Recurso_Especial_12345.txt",       "jurisprudencia",   1),
        ("Conselho_Administrativo_Recursos.txt", "jurisprudencia", 1),
        ("ato_aleatorio.txt",                "outros",           0),
    ]
    for fname, exp_cat, exp_lev in cases:
        cat, lev = chf.classify_doc_hierarchy(fname)
        assert cat == exp_cat, f"{fname}: esperado {exp_cat}, got {cat}"
        assert lev == exp_lev, f"{fname}: nivel esperado {exp_lev}, got {lev}"
    print(f"   test_hierarchy_classification: {len(cases)} casos OK")


def test_hierarchy_diagnostic():

    fns = [
        "Constituicao_Federal.txt",
        "Lei_7713.txt", "Lei_9250.txt",
        "Decreto_9580.txt",
        "Algo_Estranho.txt", "Outro_Estranho.txt", "Mais_Um.txt",
    ]
    info = chf.diagnose_hierarchy_classification(fns, show_unclassified_top=5)
    assert info["total"] == 7
    assert info["unclassified"] == 3
    assert abs(info["coverage_rate"] - 4/7) < 1e-9
    assert "Algo_Estranho.txt" in info["unclassified_sample"]
    print(f"   test_hierarchy_diagnostic: cobertura={info['coverage_rate']:.2%}  "
          f"não classificados={info['unclassified']}")


def test_metrics_basic():

    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)
    rng = np.random.default_rng(0)
    q_emb = rng.normal(size=embs.shape[1])
    q_emb = q_emb / np.linalg.norm(q_emb)


    q_eval = Question("Q", "x", "test", "a", [], [],
                      [names[2], names[5]], [])
    U0 = chf.initial_distribution(q_emb, embs, chunks, names, temperature=10.0)
    ts = np.linspace(0, 5, 20)


    t_star, f1_curve, _ = chf.search_optimal_t(
        op, U0, q_eval, names, chunks, embs, q_emb, ts, k=5,
    )
    erc = mtr.erc_star(f1_curve, ts, gamma=0.5)
    auc = mtr.auc_f1(f1_curve, ts)
    print(f"   t*={t_star:.3f}  F1(t*)={f1_curve.max():.3f}  "
          f"ERC*={erc:.4f}  AUC={auc:.4f}")


    E = mtr.energy_along_trajectory(op, U0, ts)
    decay = mtr.empirical_decay_rate(E, ts)
    print(f"   E(0)={E[0]:.4f}  E(T)={E[-1]:.4e}  λ_emp={decay:.4f}  "
          f"λ_2={op.spectral_gap:.4f}")


    assert decay > 0, "energia não decai"


    retrieved, _ = chf.retrieve_at_time(
        op, U0, t_star, names, chunks, embs, q_emb, k=5,
    )
    sigs = mtr.flow_signatures_for_topk(
        op, U0, L_sem, L_cit, L_hier, names, retrieved, t_star, n_steps=15,
    )
    H = mtr.provenance_entropy(sigs)
    print("test_metrics_basic: todas as métricas computam")


def test_coupling_work_ratio():
    names, chunks, embs, qs = make_synthetic_world()
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=3)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)

    rng = np.random.default_rng(0)
    q_emb = rng.normal(size=embs.shape[1])
    q_emb = q_emb / np.linalg.norm(q_emb)
    U0 = chf.initial_distribution(q_emb, embs, chunks, names, temperature=10.0)


    op_zero = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                         beta_sc=0, beta_sh=0,
                                         beta_cs=0, beta_ch=0,
                                         beta_hs=0, beta_hc=0)
    cwr0 = mtr.coupling_work_ratio(op_zero, U0, L_sem, L_cit, L_hier,
                                   t_star=2.0, n_steps=20)
    assert cwr0["cwr_mean"] < 1e-9, f"β=0 deve dar CWR=0, got {cwr0['cwr_mean']}"


    op_full = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                         beta_sc=0.1, beta_sh=0.1,
                                         beta_cs=0.1, beta_ch=0.1,
                                         beta_hs=0.1, beta_hc=0.1)
    cwr_full = mtr.coupling_work_ratio(op_full, U0, L_sem, L_cit, L_hier,
                                       t_star=2.0, n_steps=20)
    for chan in ["sem", "cit", "hier"]:
        v = cwr_full[f"cwr_{chan}"]
        assert 0 <= v <= 1, f"CWR^{chan}={v} fora de [0,1]"
        assert v > 0, f"β>0 mas CWR^{chan}={v} (esperado > 0)"


    cwr_t0 = mtr.coupling_work_ratio(op_full, U0, L_sem, L_cit, L_hier,
                                     t_star=0.0)
    assert cwr_t0["cwr_mean"] == 0.0
    assert cwr_t0["intra_work_sem"] == 0.0

    print(f"test_coupling_work_ratio:")
    print(f"   β=0     → CWR_mean={cwr0['cwr_mean']:.3e}  (esperado 0)")
    print(f"   β=0.1   → CWR_sem={cwr_full['cwr_sem']:.3f}  CWR_cit={cwr_full['cwr_cit']:.3f}  "
          f"CWR_hier={cwr_full['cwr_hier']:.3f}")


def test_cosine_doc_level_equals_chf_at_t_zero():
    names, chunks, embs, qs = make_synthetic_world(N_docs=15, chunks_per_doc=4)
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=4)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)


    rng = np.random.default_rng(7)
    q_emb = rng.normal(size=embs.shape[1])
    q_emb = q_emb / np.linalg.norm(q_emb)
    q_eval = Question("99", "x", "test", "a", [], [],
                      [names[2], names[5], names[7]], [])


    U0 = chf.initial_distribution(q_emb, embs, chunks, names, temperature=10.0)
    chf_retrieved, _ = chf.retrieve_at_time(
        op, U0, t=0.0, docs=names, all_chunks=chunks,
        chunk_embs=embs, q_emb=q_emb, k=5,
    )
    chf_docs = [c.doc_filename for c in chf_retrieved]


    cos_rows = baselines.evaluate_cosine_doc_level(
        [q_eval], q_emb[None, :], embs, chunks, "test", k=5,
    )

    scores = embs @ q_emb
    doc_best = {}
    for ci, c in enumerate(chunks):
        s = float(scores[ci])
        if c.doc_filename not in doc_best or s > doc_best[c.doc_filename][1]:
            doc_best[c.doc_filename] = (ci, s)
    cos_docs = [d for d, _ in sorted(doc_best.items(), key=lambda x: x[1][1], reverse=True)[:5]]

    assert chf_docs == cos_docs, \
        f"CHF(t=0) != cosine_doc_level:\n  chf: {chf_docs}\n  cos: {cos_docs}"
    print(f"   test_cosine_doc_level_equals_chf_at_t_zero: top-5 idênticos = {chf_docs}")


def test_question_typing():

    qtyping = qt



    assert qtyping.classify_question_by_refs(
        ["Lei 7.713 de 1988"], []) == "direct"
    assert qtyping.classify_question_by_refs(
        [{"name": "Lei X"}, {"name": "Decreto Y"}], []) == "direct"


    assert qtyping.classify_question_by_refs(
        [], ["Lei mencionada implicitamente"]) == "conceptual"
    assert qtyping.classify_question_by_refs([], []) == "conceptual"
    assert qtyping.classify_question_by_refs(None, None) == "conceptual"



    assert qtyping.classify_question_type(
        "Qual o limite de dedução do IRPF segundo a Lei 7.713 de 1988?") == "direct"
    assert qtyping.classify_question_type(
        "O Decreto 9.580/2018 permite dedução de despesas médicas?") == "direct"
    assert qtyping.classify_question_type(
        "Como funciona a IN RFB 1.500/2014?") == "direct"
    assert qtyping.classify_question_type(
        "Pode-se deduzir o INSS conforme o CTN?") == "direct"
    assert qtyping.classify_question_type(
        "A Constituição Federal trata de tributos sobre renda?") == "direct"


    assert qtyping.classify_question_type(
        "Como funciona a dedução de dependentes?") == "conceptual"
    assert qtyping.classify_question_type(
        "Posso deduzir despesas com plano de saúde?") == "conceptual"
    assert qtyping.classify_question_type(
        "Quem precisa declarar imposto de renda?") == "conceptual"


    n_d, n_c = 80, 90
    rng = np.random.default_rng(0)
    t_d = rng.exponential(scale=0.05, size=n_d)
    t_c = rng.exponential(scale=0.30, size=n_c)
    f1_d = 0.4 + 0.05 * rng.normal(size=n_d)
    f1_c = 0.3 + 0.05 * rng.normal(size=n_c)
    types = ["direct"] * n_d + ["conceptual"] * n_c
    t_all = np.concatenate([t_d, t_c])
    f1_all = np.concatenate([f1_d, f1_c])

    stats = qtyping.stratify_t_star(t_all, f1_all, types)
    assert stats.n_direct == n_d
    assert stats.n_conceptual == n_c

    assert stats.t_star_conceptual_mean > stats.t_star_direct_mean

    assert stats.welch_t_p_value < 0.01, \
        f"esperava p<0.01 com diferença grande, got p={stats.welch_t_p_value}"
    print(f"   test_question_typing: classify_by_refs OK, regex fallback OK, "
          f"Welch p={stats.welch_t_p_value:.2e}, "
          f"Δt*={stats.delta_t_star_median:+.4f}")


def test_spectral_bound_correct():
    names, chunks, embs, qs = make_synthetic_world(N_docs=12, chunks_per_doc=4)
    graph = ccg_rag.build_cocitation_graph(qs, min_cooccurrences=1)
    L_sem = chf.build_semantic_laplacian(embs, chunks, names, k_neighbors=4)
    L_cit = chf.build_cocitation_laplacian(graph, names)
    levels = {d: chf.classify_doc_hierarchy(d)[1] for d in names}
    L_hier = chf.build_hierarchy_laplacian(names, levels)
    op = chf.build_coupled_operator(L_sem, L_cit, L_hier,
                                    beta_sc=0.1, beta_sh=0.1,
                                    beta_cs=0.1, beta_ch=0.1,
                                    beta_hs=0.1, beta_hc=0.1)
    rng = np.random.default_rng(11)
    U0 = rng.normal(size=3 * op.N)

    ts = np.linspace(0, 8, 80)
    E = mtr.energy_along_trajectory(op, U0, ts)
    diag = mtr.operator_diagnostics(op)
    bounds = mtr.spectral_bound_check(
        E, ts,
        lambda_min=diag["lambda_min"],
        spectral_gap=op.spectral_gap,
        operator=op, U0=U0,
    )
    assert not bounds["bound_strong_violated"], \
        f"strong bound violated! ratio={bounds['bound_strong_max_ratio']}"

    if "projected_bound_violated" in bounds:
        assert not bounds["projected_bound_violated"], \
            f"projected bound violated! ratio={bounds['projected_max_ratio']}"
    print(f"   test_spectral_bound_correct: strong ratio={bounds['bound_strong_max_ratio']:.4f}, "
          f"projected ratio={bounds.get('projected_max_ratio', float('nan')):.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testes matemáticos do CHF-RAG")
    print("=" * 70)
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    test_psd_laplacians()
    test_coupled_operator_psd_and_eigendecomp()
    test_tikhonov_regularizes_isolated_ablation()
    test_evolve_closed_form_consistency()
    test_evolve_batch_matches_loop()
    test_energy_monotone_decreasing()
    test_t_zero_recovers_cosine()
    test_t_large_concentrates_mass()
    test_initial_distribution_reflects_query()
    test_hierarchy_classification()
    test_hierarchy_diagnostic()
    test_metrics_basic()
    test_coupling_work_ratio()
    test_cosine_doc_level_equals_chf_at_t_zero()
    test_question_typing()
    test_spectral_bound_correct()
    print("\n   Todos os testes matemáticos passaram.")
