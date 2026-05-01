from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

def classify_question_by_refs(
    references_explicit: List[Any],
    references_implicit: List[Any] = None,
) -> str:
    n_explicit = len(references_explicit) if references_explicit else 0
    return "direct" if n_explicit >= 1 else "conceptual"


def classify_question_by_ref_count(
    references_explicit: List[Any],
    references_implicit: List[Any] = None,
    bins: Tuple[int, int] = (1, 3),
) -> str:
    n_explicit = len(references_explicit) if references_explicit else 0
    if n_explicit == 0:
        return "sparse"
    elif n_explicit <= bins[0]:
        return "few"
    elif n_explicit <= bins[1]:
        return "medium"
    else:
        return "many"

DIPLOMA_PATTERNS = [

    re.compile(r"lei\s+(?:complementar\s+)?(?:n[ºo°]?\.?\s*)?\d{1,3}(?:[\.\,]\d{3})*"
               r"(?:\s*(?:de|/|\-)\s*\d{2,4})?",
               re.IGNORECASE),

    re.compile(r"decreto(?:-lei)?\s+(?:n[ºo°]?\.?\s*)?\d{1,3}(?:[\.\,]\d{3})*"
               r"(?:\s*(?:de|/|\-)\s*\d{2,4})?",
               re.IGNORECASE),

    re.compile(r"instru[çc][ãa]o\s+normativa(?:\s+(?:rfb|srf|srrf))?\s+"
               r"(?:n[ºo°]?\.?\s*)?\d{1,3}(?:[\.\,]\d{3})*"
               r"(?:\s*(?:de|/|\-)\s*\d{2,4})?",
               re.IGNORECASE),

    re.compile(r"\bIN\s+(?:RFB\s+)?(?:n[ºo°]?\.?\s*)?\d{1,3}(?:[\.\,]\d{3})*",
               re.IGNORECASE),

    re.compile(r"constitui[çc][ãa]o(?:\s+federal)?", re.IGNORECASE),

    re.compile(r"\bCTN\b|c[oó]digo\s+tribut[áa]rio", re.IGNORECASE),

    re.compile(r"portaria\s+(?:n[ºo°]?\.?\s*)?\d+", re.IGNORECASE),
    re.compile(r"resolu[çc][ãa]o\s+(?:n[ºo°]?\.?\s*)?\d+", re.IGNORECASE),
    re.compile(r"ato\s+declarat[óo]rio", re.IGNORECASE),

    re.compile(r"medida\s+provis[óo]ria\s+(?:n[ºo°]?\.?\s*)?\d+", re.IGNORECASE),

    re.compile(r"emenda\s+constitucional\s+(?:n[ºo°]?\.?\s*)?\d+", re.IGNORECASE),
]


def count_diploma_mentions(question_text: str) -> int:

    n = 0
    for pat in DIPLOMA_PATTERNS:
        n += len(pat.findall(question_text))
    return n


def classify_question_type(question_text: str) -> str:
    n = count_diploma_mentions(question_text)
    return "direct" if n >= 1 else "conceptual"


@dataclass
class StratifiedTStarStats:

    n_direct: int
    n_conceptual: int
    t_star_direct_median: float
    t_star_direct_mean: float
    t_star_conceptual_median: float
    t_star_conceptual_mean: float
    f1_direct_mean: float
    f1_conceptual_mean: float
    delta_t_star_median: float
    welch_t_p_value: float


def stratify_t_star(
    t_stars: np.ndarray,
    f1_at_t_star: np.ndarray,
    question_types: List[str],
) -> StratifiedTStarStats:
    types = np.array(question_types)
    direct_mask = types == "direct"
    conc_mask = types == "conceptual"

    if not direct_mask.any() or not conc_mask.any():

        return StratifiedTStarStats(
            n_direct=int(direct_mask.sum()),
            n_conceptual=int(conc_mask.sum()),
            t_star_direct_median=float("nan"),
            t_star_direct_mean=float("nan"),
            t_star_conceptual_median=float("nan"),
            t_star_conceptual_mean=float("nan"),
            f1_direct_mean=float("nan"),
            f1_conceptual_mean=float("nan"),
            delta_t_star_median=float("nan"),
            welch_t_p_value=float("nan"),
        )

    t_d = t_stars[direct_mask]
    t_c = t_stars[conc_mask]
    f1_d = f1_at_t_star[direct_mask]
    f1_c = f1_at_t_star[conc_mask]




    p_value = _welch_t_p_value(t_d, t_c)

    return StratifiedTStarStats(
        n_direct=int(direct_mask.sum()),
        n_conceptual=int(conc_mask.sum()),
        t_star_direct_median=float(np.median(t_d)),
        t_star_direct_mean=float(np.mean(t_d)),
        t_star_conceptual_median=float(np.median(t_c)),
        t_star_conceptual_mean=float(np.mean(t_c)),
        f1_direct_mean=float(np.mean(f1_d)),
        f1_conceptual_mean=float(np.mean(f1_c)),
        delta_t_star_median=float(np.median(t_c) - np.median(t_d)),
        welch_t_p_value=p_value,
    )


def _welch_t_p_value(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 1.0 if abs(float(a.mean()) - float(b.mean())) < 1e-12 else 0.0
    t_stat = (float(a.mean()) - float(b.mean())) / se



    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    return float(p)
