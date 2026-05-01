from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_loader import LegalChunk, Question
from .ccg_rag import CoCitationGraph

logger = logging.getLogger(__name__)

HIERARCHY_LEVELS = {
    "constituicao":     5,
    "emenda":           5,
    "tratado":          5,
    "lei_complementar": 4,
    "lei":              3,
    "decreto_lei":      3,
    "medida_provisoria": 3,
    "decreto":          2,
    "portaria":         1,
    "instrucao":        1,
    "circular":         1,
    "resolucao":        1,
    "jurisprudencia":   1,
    "outros":           0,
}

_HIERARCHY_PATTERNS: List[Tuple[str, re.Pattern]] = [

    ("jurisprudencia",   re.compile(
        r"acord[ãa]o|\bcarf\b|conselho[_\s\-]administrativo|"
        r"\bstf\b|\bstj\b|\btst\b|\btrf\b|"
        r"s[ué]mula[_\s\-]vinculante|s[ué]mula[_\s\-]\d|"
        r"recurso[_\s\-]extraordin|recurso[_\s\-]especial|"
        r"agravo|apela[çc][ãa]o|mandado[_\s\-]seguran|habeas|"
        r"jurispruden|julgado|precedente",
        re.IGNORECASE)),

    ("constituicao",     re.compile(r"constitui[çc][ãa]o|cf[_\s\-]?(?:19)?88|carta[_\s\-]magna",
                                    re.IGNORECASE)),
    ("emenda",           re.compile(r"emenda[_\s\-]constituc|\bec[_\s\-]\d|emenda[_\s\-]\d",
                                    re.IGNORECASE)),
    ("tratado",          re.compile(r"tratado|conven[çc][ãa]o[_\s\-]internacional|acordo[_\s\-]bilateral",
                                    re.IGNORECASE)),

    ("lei_complementar", re.compile(r"lei[_\s\-]?complementar|^lc[_\s\-]\d|\blcp?[_\s\-]?\d{1,4}",
                                    re.IGNORECASE)),

    ("decreto_lei",      re.compile(r"decreto[_\s\-]?lei|\bdl[_\s\-]\d|d[_\s\-]?lei[_\s\-]?\d",
                                    re.IGNORECASE)),

    ("medida_provisoria", re.compile(r"medida[_\s\-]provisoria|medida[_\s\-]provis[óo]ria|^mp[_\s\-]?\d|\bmpv[_\s\-]?\d",
                                     re.IGNORECASE)),

    ("decreto",          re.compile(r"^decreto|decreto[_\s\-]?n?[_\s\-]?\d|^dec[_\s\-]\d|\bdec\b[_\s\-]?\d",
                                    re.IGNORECASE)),

    ("instrucao",        re.compile(
        r"instru[çc][ãa]o[_\s\-]normativa|^in[_\s\-](?:rfb|srf|srrf)?[_\s\-]?\d|"
        r"ato[_\s\-]declarat|solu[çc][ãa]o[_\s\-]de?[_\s\-]consulta|"
        r"\bsc[_\s\-]\d{2,}|parecer[_\s\-]normativo|^pn[_\s\-]\d",
        re.IGNORECASE)),

    ("portaria",         re.compile(r"portaria|^port[_\s\-]\d", re.IGNORECASE)),

    ("resolucao",        re.compile(r"resolu[çc][ãa]o|^res[_\s\-]\d", re.IGNORECASE)),

    ("circular",         re.compile(r"circular|comunicado|nota[_\s\-]t[ée]cnica", re.IGNORECASE)),

    ("lei",              re.compile(r"\blei\b|^lei[_\s\-]?n?[_\s\-]?\d|\blei_\d|^l[_\s\-]\d{4,}",
                                    re.IGNORECASE)),

    ("lei_complementar", re.compile(r"\bctn\b|c[oó]digo[_\s\-]tribut[áa]rio", re.IGNORECASE)),
]


def classify_doc_hierarchy(filename: str) -> Tuple[str, int]:

    for cat, pat in _HIERARCHY_PATTERNS:
        if pat.search(filename):
            return cat, HIERARCHY_LEVELS[cat]
    return "outros", 0


def diagnose_hierarchy_classification(
    filenames: List[str],
    show_unclassified_top: int = 30,
) -> Dict[str, any]:
    counts: Dict[str, int] = {k: 0 for k in HIERARCHY_LEVELS}
    unclassified: List[str] = []

    for fn in filenames:
        cat, _ = classify_doc_hierarchy(fn)
        counts[cat] = counts.get(cat, 0) + 1
        if cat == "outros":
            unclassified.append(fn)

    total = len(filenames)
    classified = total - counts.get("outros", 0)
    coverage = classified / total if total else 0.0


    sample = unclassified[:show_unclassified_top]

    info = {
        "total": total,
        "classified": classified,
        "unclassified": counts.get("outros", 0),
        "coverage_rate": coverage,
        "counts_by_category": counts,
        "unclassified_sample": sample,
    }
    return info


def build_semantic_laplacian(
    chunk_embs: np.ndarray,
    all_chunks: List[LegalChunk],
    docs: List[str],
    k_neighbors: int = 10,
) -> np.ndarray:
    N = len(docs)
    doc_idx = {d: i for i, d in enumerate(docs)}


    doc_to_chunk_idx: Dict[str, List[int]] = {d: [] for d in docs}
    for i, c in enumerate(all_chunks):
        if c.doc_filename in doc_to_chunk_idx:
            doc_to_chunk_idx[c.doc_filename].append(i)

    logger.info("Construindo similaridade documento-documento (N=%d) ...", N)
    sim_doc = np.zeros((N, N), dtype=np.float32)

    for i, di in enumerate(docs):
        ci = doc_to_chunk_idx[di]
        if not ci:
            continue
        Ei = chunk_embs[ci]
        for j, dj in enumerate(docs):
            if j <= i:
                continue
            cj = doc_to_chunk_idx[dj]
            if not cj:
                continue
            Ej = chunk_embs[cj]

            block = Ei @ Ej.T
            top = min(3, block.size)
            if top == 0:
                continue
            top_vals = np.partition(block.ravel(), -top)[-top:]
            sim_doc[i, j] = sim_doc[j, i] = float(top_vals.mean())


    W = np.zeros_like(sim_doc)
    for i in range(N):
        if sim_doc[i].sum() == 0:
            continue

        idx = np.argsort(sim_doc[i])[::-1][:k_neighbors]
        for j in idx:
            if i == j or sim_doc[i, j] <= 0:
                continue
            W[i, j] = W[j, i] = sim_doc[i, j]

    L = _normalized_laplacian(W)
    logger.info("L^sem: shape=%s | nnz/N=%.1f", L.shape, np.count_nonzero(W) / N)
    return L


def build_cocitation_laplacian(
    graph: CoCitationGraph,
    docs: List[str],
) -> np.ndarray:
    N = len(docs)
    doc_idx = {d: i for i, d in enumerate(docs)}
    W = np.zeros((N, N), dtype=np.float32)

    for d_i, neighbors in graph.cocit_matrix.items():
        if d_i not in doc_idx:
            continue
        i = doc_idx[d_i]
        for d_j, w in neighbors.items():
            if d_j not in doc_idx:
                continue
            j = doc_idx[d_j]
            if i == j:
                continue
            W[i, j] = float(w)
            W[j, i] = float(w)

    L = _normalized_laplacian(W)
    logger.info("L^cit: shape=%s | nnz/N=%.1f", L.shape, np.count_nonzero(W) / N)
    return L


def build_hierarchy_laplacian(
    docs: List[str],
    doc_levels: Dict[str, int],
    coupling_strength: float = 1.0,
) -> np.ndarray:
    N = len(docs)
    levels = np.array([doc_levels.get(d, 0) for d in docs], dtype=np.int32)

    W = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            dlevel = abs(int(levels[i]) - int(levels[j]))
            if dlevel <= 1 and (levels[i] > 0 or levels[j] > 0):

                w = float(np.exp(-dlevel)) * coupling_strength
                W[i, j] = W[j, i] = w

    L = _normalized_laplacian(W)


    unique_levels = sorted(set(int(v) for v in levels.tolist()))
    n_per_level = {f"nivel_{lv}": int((levels == lv).sum())
                   for lv in unique_levels if lv > 0}
    n_isolated = int((levels == 0).sum())
    logger.info(
        "L^hier: shape=%s | docs por nível=%s | isolados(nível 0)=%d",
        L.shape, n_per_level, n_isolated,
    )
    return L


def _normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(W.shape[0], dtype=W.dtype) - D_inv_sqrt @ W @ D_inv_sqrt

    L = 0.5 * (L + L.T)
    return L




@dataclass
class CoupledOperator:
    M: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    N: int
    betas: Tuple[float, float, float, float, float, float]

    spectral_gap: float

    def heat_kernel(self, t: float) -> np.ndarray:
        return self.eigvecs * np.exp(-t * self.eigvals)[None, :] @ self.eigvecs.T

    def evolve(self, U0: np.ndarray, t: float) -> np.ndarray:
        alpha = self.eigvecs.T @ U0
        return self.eigvecs @ (np.exp(-t * self.eigvals) * alpha)

    def evolve_batch(self, U0: np.ndarray, ts: np.ndarray) -> np.ndarray:
        alpha = self.eigvecs.T @ U0
        decay = np.exp(-ts[:, None] * self.eigvals[None, :])
        coeffs = decay * alpha[None, :]

        return coeffs @ self.eigvecs.T


def build_coupled_operator(
    L_sem: np.ndarray,
    L_cit: np.ndarray,
    L_hier: np.ndarray,
    beta_sc: float = 0.1,
    beta_sh: float = 0.1,
    beta_cs: float = 0.1,
    beta_ch: float = 0.1,
    beta_hs: float = 0.1,
    beta_hc: float = 0.1,
    tikhonov: float = 1e-3,
) -> CoupledOperator:
    N = L_sem.shape[0]
    assert L_cit.shape == (N, N) and L_hier.shape == (N, N), \
        "Os três Laplacianos precisam ter o mesmo número de nós."

    I = np.eye(N, dtype=L_sem.dtype)
    M = np.zeros((3 * N, 3 * N), dtype=np.float64)


    M[:N, :N]           = L_sem  + (beta_sc + beta_sh) * I
    M[N:2*N, N:2*N]     = L_cit  + (beta_cs + beta_ch) * I
    M[2*N:, 2*N:]       = L_hier + (beta_hs + beta_hc) * I


    M[:N, N:2*N]        = -beta_sc * I
    M[:N, 2*N:]         = -beta_sh * I
    M[N:2*N, :N]        = -beta_cs * I
    M[N:2*N, 2*N:]      = -beta_ch * I
    M[2*N:, :N]         = -beta_hs * I
    M[2*N:, N:2*N]      = -beta_hc * I


    M = 0.5 * (M + M.T)
    if tikhonov > 0:
        M = M + tikhonov * np.eye(3 * N, dtype=M.dtype)
    logger.info("Diagonalizando M (3N × 3N = %d × %d) ...", 3 * N, 3 * N)
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    spectral_gap = float(eigvals[1] - eigvals[0])

    min_eig = float(eigvals[0])
    if min_eig < -1e-6:
        logger.warning("M não é PSD: λ_min = %.3e (esperado ≥ 0)", min_eig)
    else:
        logger.info("M PSD ok: λ_min = %.3e | λ_2 (gap) = %.4f | tikhonov=%.0e",
                    min_eig, spectral_gap, tikhonov)

    return CoupledOperator(
        M=M,
        eigvals=eigvals,
        eigvecs=eigvecs,
        N=N,
        betas=(beta_sc, beta_sh, beta_cs, beta_ch, beta_hs, beta_hc),
        spectral_gap=spectral_gap,
    )

def initial_distribution(
    q_emb: np.ndarray,
    chunk_embs: np.ndarray,
    all_chunks: List[LegalChunk],
    docs: List[str],
    temperature: float = 10.0,
) -> np.ndarray:

    cos_scores = chunk_embs @ q_emb
    doc_idx = {d: i for i, d in enumerate(docs)}
    N = len(docs)

    doc_max = -np.inf * np.ones(N, dtype=np.float32)
    for c, sc in zip(all_chunks, cos_scores):
        i = doc_idx.get(c.doc_filename)
        if i is not None and sc > doc_max[i]:
            doc_max[i] = float(sc)

    doc_max[doc_max == -np.inf] = doc_max[doc_max != -np.inf].min() - 1.0

    z = doc_max - doc_max.max()
    p = np.exp(temperature * z)
    p = p / p.sum()


    return np.concatenate([p, p, p]).astype(np.float64)


def score_at_time(
    operator: CoupledOperator,
    U0: np.ndarray,
    t: float,
    w_s: float = 1/3,
    w_c: float = 1/3,
    w_h: float = 1/3,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    Ut = operator.evolve(U0, t)
    N = operator.N
    u_sem  = Ut[:N]
    u_cit  = Ut[N:2*N]
    u_hier = Ut[2*N:]
    s = w_s * u_sem + w_c * u_cit + w_h * u_hier
    return s, {"sem": u_sem, "cit": u_cit, "hier": u_hier}


def channel_flow(
    operator: CoupledOperator,
    U0: np.ndarray,
    t: float,
    L_sem: np.ndarray,
    L_cit: np.ndarray,
    L_hier: np.ndarray,
) -> Dict[str, np.ndarray]:
    N = operator.N
    Ut = operator.evolve(U0, t)
    u_sem, u_cit, u_hier = Ut[:N], Ut[N:2*N], Ut[2*N:]
    bsc, bsh, bcs, bch, bhs, bhc = operator.betas

    flow_sem  = -L_sem  @ u_sem  + bsc * (u_cit  - u_sem)  + bsh * (u_hier - u_sem)
    flow_cit  = -L_cit  @ u_cit  + bcs * (u_sem  - u_cit)  + bch * (u_hier - u_cit)
    flow_hier = -L_hier @ u_hier + bhs * (u_sem  - u_hier) + bhc * (u_cit  - u_hier)

    return {"sem": flow_sem, "cit": flow_cit, "hier": flow_hier}




def retrieve_at_time(
    operator: CoupledOperator,
    U0: np.ndarray,
    t: float,
    docs: List[str],
    all_chunks: List[LegalChunk],
    chunk_embs: np.ndarray,
    q_emb: np.ndarray,
    k: int = 10,
    w_s: float = 1/3,
    w_c: float = 1/3,
    w_h: float = 1/3,
) -> Tuple[List[LegalChunk], Dict[str, np.ndarray]]:
    s, channels = score_at_time(operator, U0, t, w_s, w_c, w_h)
    doc_order = np.argsort(s)[::-1]

    cos_scores = chunk_embs @ q_emb
    doc_to_chunks: Dict[str, List[int]] = {}
    for i, c in enumerate(all_chunks):
        doc_to_chunks.setdefault(c.doc_filename, []).append(i)

    retrieved: List[LegalChunk] = []
    seen_docs = set()
    for di in doc_order:
        d = docs[di]
        if d in seen_docs:
            continue
        seen_docs.add(d)
        chunk_ids = doc_to_chunks.get(d, [])
        if not chunk_ids:
            continue

        best = max(chunk_ids, key=lambda j: cos_scores[j])
        retrieved.append(all_chunks[best])
        if len(retrieved) >= k:
            break

    return retrieved, channels


def search_optimal_t(
    operator: CoupledOperator,
    U0: np.ndarray,
    question: Question,
    docs: List[str],
    all_chunks: List[LegalChunk],
    chunk_embs: np.ndarray,
    q_emb: np.ndarray,
    ts: np.ndarray,
    k: int = 10,
    w_s: float = 1/3,
    w_c: float = 1/3,
    w_h: float = 1/3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    from .baselines import ir_metrics

    Ut_batch = operator.evolve_batch(U0, ts)
    N = operator.N

    cos_scores = chunk_embs @ q_emb
    doc_to_chunks: Dict[str, List[int]] = {}
    for i, c in enumerate(all_chunks):
        doc_to_chunks.setdefault(c.doc_filename, []).append(i)

    f1_curve = np.zeros(len(ts), dtype=np.float64)
    for ti, _t in enumerate(ts):
        Ut = Ut_batch[ti]
        s = w_s * Ut[:N] + w_c * Ut[N:2*N] + w_h * Ut[2*N:]
        order = np.argsort(s)[::-1]

        retrieved, seen = [], set()
        for di in order:
            d = docs[di]
            if d in seen:
                continue
            seen.add(d)
            cids = doc_to_chunks.get(d, [])
            if not cids:
                continue
            best = max(cids, key=lambda j: cos_scores[j])
            retrieved.append(all_chunks[best])
            if len(retrieved) >= k:
                break

        m = ir_metrics(retrieved, question)
        f1_curve[ti] = m["f1_k"]

    t_star_idx = int(np.argmax(f1_curve))
    return float(ts[t_star_idx]), f1_curve, ts

def check_psd(L: np.ndarray, name: str = "L", tol: float = 1e-6) -> Dict[str, float]:

    w = np.linalg.eigvalsh(L)
    info = {
        "name": name,
        "lambda_min": float(w[0]),
        "lambda_max": float(w[-1]),
        "psd": bool(w[0] > -tol),
    }
    logger.debug("%s: λ_min=%.3e λ_max=%.3e PSD=%s",
                 name, info["lambda_min"], info["lambda_max"], info["psd"])
    return info


def check_mass_conservation(
    operator: CoupledOperator,
    U0: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    Ut_batch = operator.evolve_batch(U0, ts)
    masses = Ut_batch.sum(axis=1)
    return masses
