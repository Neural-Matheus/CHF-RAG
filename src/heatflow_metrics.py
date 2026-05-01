from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_trapz = getattr(np, 'trapezoid', None) or np.trapz

from .data_loader import LegalChunk, Question
from .coupled_heatflow import (
    CoupledOperator,
    channel_flow,
    score_at_time,
)

logger = logging.getLogger(__name__)

def erc_star(
    f1_curve: np.ndarray,
    ts: np.ndarray,
    gamma: float = 0.5,
) -> float:

    weights = np.exp(-gamma * ts)
    integrand = f1_curve * weights
    return float(_trapz(integrand, ts))


def auc_f1(f1_curve: np.ndarray, ts: np.ndarray) -> float:

    return float(_trapz(f1_curve, ts))


@dataclass
class TStarStats:
    t_star_per_question: np.ndarray
    f1_at_t_star: np.ndarray
    f1_at_t_zero: np.ndarray
    delta_f1: np.ndarray
    n_improved: int
    median_t_star: float
    iqr_t_star: Tuple[float, float]


def aggregate_t_star(
    f1_curves: np.ndarray,
    ts: np.ndarray,
) -> TStarStats:

    Q, T = f1_curves.shape
    t_star_idx = np.argmax(f1_curves, axis=1)
    t_star = ts[t_star_idx]
    f1_star = f1_curves[np.arange(Q), t_star_idx]
    f1_zero = f1_curves[:, 0]
    delta = f1_star - f1_zero

    return TStarStats(
        t_star_per_question=t_star,
        f1_at_t_star=f1_star,
        f1_at_t_zero=f1_zero,
        delta_f1=delta,
        n_improved=int((delta > 0).sum()),
        median_t_star=float(np.median(t_star)),
        iqr_t_star=(float(np.quantile(t_star, 0.25)),
                    float(np.quantile(t_star, 0.75))),
    )

def energy_along_trajectory(
    operator: CoupledOperator,
    U0: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:

    Ut = operator.evolve_batch(U0, ts)

    MU = Ut @ operator.M.T
    E = 0.5 * np.einsum("ti,ti->t", Ut, MU)
    return E


def empirical_decay_rate(E: np.ndarray, ts: np.ndarray) -> float:

    valid = (E > 1e-12) & (np.arange(len(E)) > 0)
    if valid.sum() < 5:
        return float("nan")
    log_E = np.log(E[valid])
    t_v = ts[valid]

    A = np.vstack([t_v, np.ones_like(t_v)]).T
    coefs, *_ = np.linalg.lstsq(A, log_E, rcond=None)
    lambda_emp = -float(coefs[0])
    return lambda_emp


def spectral_bound_check(
    E: np.ndarray,
    ts: np.ndarray,
    lambda_min: float,
    spectral_gap: float,
    operator: Optional["CoupledOperator"] = None,
    U0: Optional[np.ndarray] = None,
) -> Dict[str, float]:

    out = {
        "lambda_min":       float(lambda_min),
        "spectral_gap":     float(spectral_gap),
    }

    bound_strong = E[0] * np.exp(-2 * lambda_min * ts)
    violation_strong = bool((E > bound_strong + 1e-9 * abs(E[0])).any())
    ratio_strong = E / np.where(bound_strong > 1e-12, bound_strong, 1e-12)
    out["bound_strong_violated"] = violation_strong
    out["bound_strong_max_ratio"] = float(np.max(ratio_strong))

    bound_opt = E[0] * np.exp(-2 * spectral_gap * ts)
    violation_opt = bool((E > bound_opt + 1e-9 * abs(E[0])).any())
    out["bound_optimistic_violated"] = violation_opt
    ratio_opt = E / np.where(bound_opt > 1e-12, bound_opt, 1e-12)
    out["bound_optimistic_max_ratio"] = float(np.max(ratio_opt))


    if operator is not None and U0 is not None:


        ker_mask = np.abs(operator.eigvals) < 1e-9
        if ker_mask.any():
            V_ker = operator.eigvecs[:, ker_mask]
            U_inf = V_ker @ (V_ker.T @ U0)
        else:
            U_inf = np.zeros_like(U0)

        Ut = operator.evolve_batch(U0, ts)
        diff = Ut - U_inf[None, :]

        Mdiff = diff @ operator.M.T
        E_proj = 0.5 * np.einsum("ti,ti->t", diff, Mdiff)
        E_proj = np.maximum(E_proj, 0)


        bound_proj = E_proj[0] * np.exp(-2 * spectral_gap * ts)
        viol_proj = bool((E_proj > bound_proj + 1e-9 * abs(E_proj[0] + 1e-12)).any())
        out["projected_bound_violated"] = viol_proj
        out["projected_max_ratio"] = float(np.max(
            E_proj / np.where(bound_proj > 1e-12, bound_proj, 1e-12)
        ))

    return out



def cheeger_bound_check(
    E: np.ndarray, ts: np.ndarray, spectral_gap: float,
) -> Dict[str, float]:

    bound = E[0] * np.exp(-spectral_gap * ts)
    violation = bool((E > bound + 1e-9 * abs(E[0])).any())
    ratio = E / np.where(bound > 1e-12, bound, 1e-12)
    return {
        "spectral_gap": float(spectral_gap),
        "bound_violated": violation,
        "max_ratio": float(np.max(ratio)),
        "median_ratio": float(np.median(ratio)),
    }


@dataclass
class FlowSignature:

    doc_filename: str
    chunk_id: int
    flow_sem: float
    flow_cit: float
    flow_hier: float

    rel_sem: float
    rel_cit: float
    rel_hier: float


def integrate_channel_flow(
    operator: CoupledOperator,
    U0: np.ndarray,
    L_sem: np.ndarray,
    L_cit: np.ndarray,
    L_hier: np.ndarray,
    docs: List[str],
    doc_indices_in_topk: List[int],
    t_star: float,
    n_steps: int = 30,
) -> List[Tuple[int, float, float, float]]:
    if t_star <= 0:

        return [(di, 0.0, 0.0, 0.0) for di in doc_indices_in_topk]

    ts = np.linspace(0.0, t_star, n_steps)

    Ut_batch = operator.evolve_batch(U0, ts)
    N = operator.N
    bsc, bsh, bcs, bch, bhs, bhc = operator.betas

    flows_sem = np.zeros((n_steps, N))
    flows_cit = np.zeros((n_steps, N))
    flows_hier = np.zeros((n_steps, N))

    for ti in range(n_steps):
        u_sem  = Ut_batch[ti, :N]
        u_cit  = Ut_batch[ti, N:2*N]
        u_hier = Ut_batch[ti, 2*N:]
        flows_sem[ti]  = -L_sem  @ u_sem  + bsc * (u_cit  - u_sem)  + bsh * (u_hier - u_sem)
        flows_cit[ti]  = -L_cit  @ u_cit  + bcs * (u_sem  - u_cit)  + bch * (u_hier - u_cit)
        flows_hier[ti] = -L_hier @ u_hier + bhs * (u_sem  - u_hier) + bhc * (u_cit  - u_hier)


    int_sem  = _trapz(np.abs(flows_sem),  ts, axis=0)
    int_cit  = _trapz(np.abs(flows_cit),  ts, axis=0)
    int_hier = _trapz(np.abs(flows_hier), ts, axis=0)

    return [(di,
             float(int_sem[di]),
             float(int_cit[di]),
             float(int_hier[di])) for di in doc_indices_in_topk]


def flow_signatures_for_topk(
    operator: CoupledOperator,
    U0: np.ndarray,
    L_sem: np.ndarray,
    L_cit: np.ndarray,
    L_hier: np.ndarray,
    docs: List[str],
    retrieved_chunks: List[LegalChunk],
    t_star: float,
    n_steps: int = 30,
) -> List[FlowSignature]:

    doc_idx = {d: i for i, d in enumerate(docs)}
    indices = [doc_idx[c.doc_filename] for c in retrieved_chunks
               if c.doc_filename in doc_idx]

    raw = integrate_channel_flow(
        operator, U0, L_sem, L_cit, L_hier,
        docs, indices, t_star, n_steps,
    )

    sigs = []
    for c, (di, fs, fc, fh) in zip(retrieved_chunks, raw):
        total = fs + fc + fh
        if total <= 0:
            rs, rc, rh = 1/3, 1/3, 1/3
        else:
            rs, rc, rh = fs / total, fc / total, fh / total
        sigs.append(FlowSignature(
            doc_filename=c.doc_filename,
            chunk_id=c.chunk_id,
            flow_sem=fs, flow_cit=fc, flow_hier=fh,
            rel_sem=rs, rel_cit=rc, rel_hier=rh,
        ))
    return sigs


def provenance_entropy(signatures: List[FlowSignature]) -> float:
    if not signatures:
        return 0.0
    rs = np.mean([s.rel_sem for s in signatures])
    rc = np.mean([s.rel_cit for s in signatures])
    rh = np.mean([s.rel_hier for s in signatures])
    p = np.array([rs, rc, rh])
    p = p / p.sum() if p.sum() > 0 else np.array([1/3, 1/3, 1/3])
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())




def coupling_work_ratio(
    operator: CoupledOperator,
    U0: np.ndarray,
    L_sem: np.ndarray,
    L_cit: np.ndarray,
    L_hier: np.ndarray,
    t_star: float,
    n_steps: int = 30,
) -> Dict[str, float]:
    if t_star <= 0:
        return {
            "cwr_sem": 0.0, "cwr_cit": 0.0, "cwr_hier": 0.0,
            "cwr_mean": 0.0, "cwr_max": 0.0,
            "intra_work_sem": 0.0, "cross_work_sem": 0.0,
            "intra_work_cit": 0.0, "cross_work_cit": 0.0,
            "intra_work_hier": 0.0, "cross_work_hier": 0.0,
        }

    ts = np.linspace(0.0, t_star, n_steps)
    Ut_batch = operator.evolve_batch(U0, ts)
    N = operator.N
    bsc, bsh, bcs, bch, bhs, bhc = operator.betas

    intra_norm  = {"sem": np.zeros(n_steps), "cit": np.zeros(n_steps), "hier": np.zeros(n_steps)}
    cross_norm  = {"sem": np.zeros(n_steps), "cit": np.zeros(n_steps), "hier": np.zeros(n_steps)}

    for ti in range(n_steps):
        u_sem  = Ut_batch[ti, :N]
        u_cit  = Ut_batch[ti, N:2*N]
        u_hier = Ut_batch[ti, 2*N:]

        intra_sem  = -L_sem  @ u_sem
        intra_cit  = -L_cit  @ u_cit
        intra_hier = -L_hier @ u_hier

        cross_sem  = bsc * (u_cit  - u_sem)  + bsh * (u_hier - u_sem)
        cross_cit  = bcs * (u_sem  - u_cit)  + bch * (u_hier - u_cit)
        cross_hier = bhs * (u_sem  - u_hier) + bhc * (u_cit  - u_hier)

        intra_norm["sem"][ti]   = float(np.linalg.norm(intra_sem))
        intra_norm["cit"][ti]   = float(np.linalg.norm(intra_cit))
        intra_norm["hier"][ti]  = float(np.linalg.norm(intra_hier))
        cross_norm["sem"][ti]   = float(np.linalg.norm(cross_sem))
        cross_norm["cit"][ti]   = float(np.linalg.norm(cross_cit))
        cross_norm["hier"][ti]  = float(np.linalg.norm(cross_hier))

    out = {}
    cwrs = []
    for chan in ["sem", "cit", "hier"]:
        intra_work = float(_trapz(intra_norm[chan], ts))
        cross_work = float(_trapz(cross_norm[chan], ts))
        total = intra_work + cross_work
        cwr = (cross_work / total) if total > 1e-12 else 0.0
        out[f"intra_work_{chan}"] = intra_work
        out[f"cross_work_{chan}"] = cross_work
        out[f"cwr_{chan}"]        = cwr
        cwrs.append(cwr)

    out["cwr_mean"] = float(np.mean(cwrs))
    out["cwr_max"]  = float(np.max(cwrs))
    return out


def coupling_lift(
    f1_coupled: np.ndarray,
    f1_isolated: Dict[str, np.ndarray],
) -> Dict[str, float]:
    lifts = {}
    for name, f1_iso in f1_isolated.items():
        diff = f1_coupled - f1_iso
        lifts[f"lift_vs_{name}"] = float(np.mean(diff))
        lifts[f"lift_vs_{name}_pct_improved"] = float((diff > 0).mean())

    if f1_isolated:
        max_iso = np.max(np.stack(list(f1_isolated.values()), axis=0), axis=0)
        diff = f1_coupled - max_iso
        lifts["lift_vs_best_isolated"] = float(np.mean(diff))
        lifts["lift_vs_best_isolated_pct_improved"] = float((diff > 0).mean())
    return lifts

def operator_diagnostics(operator: CoupledOperator) -> Dict[str, float]:
    eigs = operator.eigvals

    return {
        "lambda_min": float(eigs[0]),
        "lambda_2": float(eigs[1]),
        "lambda_max": float(eigs[-1]),
        "spectral_gap": float(operator.spectral_gap),
        "n_zero_eigs": int((np.abs(eigs) < 1e-9).sum()),
        "is_psd": bool(eigs[0] > -1e-6),
    }
