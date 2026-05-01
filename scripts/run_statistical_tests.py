import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


OUTPUT_DIR = Path("/app/outputs/statistical_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHF_CSV    = Path("/app/outputs/coupled_heatflow/chf_raw.csv")
HYBRID_CSV = Path("/app/outputs/hybrid_rerank/hybrid_raw.csv")


def welch_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan")
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 0.0, 1.0
    t_stat = (float(a.mean()) - float(b.mean())) / se

    if var_a < 1e-12 and var_b < 1e-12:
        return 0.0, 1.0
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2 / max(n_a - 1, 1) +
              (var_b / n_b) ** 2 / max(n_b - 1, 1))
    df = df_num / df_den if df_den > 0 else (n_a + n_b - 2)

    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    except ImportError:
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    return float(t_stat), float(p)


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), f"Tamanhos diferentes: {len(a)} vs {len(b)}"
    diff = a - b
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), n
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1)) / np.sqrt(n)
    if se < 1e-12:
        return 0.0, 1.0, n
    t_stat = mean_diff / se
    df = n - 1
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    except ImportError:
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    return float(t_stat), float(p), int(n)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[diff != 0]
    n = len(diff)
    if n < 6:
        return float("nan"), float("nan"), n
    try:
        from scipy import stats
        result = stats.wilcoxon(diff)
        return float(result.statistic), float(result.pvalue), int(n)
    except ImportError:

        abs_diff = np.abs(diff)
        ranks = pd.Series(abs_diff).rank().values
        W_pos = float(np.sum(ranks[diff > 0]))

        mean_W = n * (n + 1) / 4
        var_W = n * (n + 1) * (2 * n + 1) / 24
        z = (W_pos - mean_W) / np.sqrt(var_W)
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
        return float(W_pos), float(p), int(n)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    if pooled_var < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def cohens_d_paired(diff: np.ndarray) -> float:

    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(diff.mean() / sd)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    n = len(p_values)
    sorted_idx = sorted(range(n), key=lambda i: p_values[i])
    reject = [False] * n
    for rank, idx in enumerate(sorted_idx):
        threshold = alpha / (n - rank)
        if p_values[idx] < threshold:
            reject[idx] = True
        else:
            break
    return reject


def significance_label(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def effect_size_label(d: float) -> str:

    abs_d = abs(d)
    if abs_d < 0.2:
        return "trivial"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def chf_paired_tests(df_chf: pd.DataFrame) -> pd.DataFrame:
    chf_strategy = "chf_full_w_struct_heavy"
    baselines = ["cosine_doc_level", "cosine", "sem_only", "cit_only",
                 "hier_only", "sem_cit"]

    rows = []
    for emb in df_chf["embedder"].unique():
        sub = df_chf[df_chf["embedder"] == emb]
        chf = sub[sub["strategy"] == chf_strategy].sort_values("question_number")
        if not len(chf):
            continue

        for baseline in baselines:
            base = sub[sub["strategy"] == baseline].sort_values("question_number")
            if not len(base):
                continue

            merged = pd.merge(
                chf[["question_number", "f1_k", "recall_k", "precision_k"]],
                base[["question_number", "f1_k", "recall_k", "precision_k"]],
                on="question_number", suffixes=("_chf", "_base")
            )
            if len(merged) < 10:
                continue

            for metric in ["f1_k", "recall_k", "precision_k"]:
                a = merged[f"{metric}_chf"].values
                b = merged[f"{metric}_base"].values
                t_stat, p_paired, n = paired_t_test(a, b)
                w_stat, p_wilcox, n_nz = wilcoxon_signed_rank(a, b)
                d_paired = cohens_d_paired(a - b)
                rows.append({
                    "embedder": emb,
                    "comparison": f"{chf_strategy} vs {baseline}",
                    "metric": metric,
                    "n_pairs": n,
                    "mean_chf": float(a.mean()),
                    "mean_base": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "t_stat": t_stat,
                    "p_paired_t": p_paired,
                    "p_wilcoxon": p_wilcox,
                    "cohens_d": d_paired,
                    "effect_size": effect_size_label(d_paired),
                    "sig_paired_t": significance_label(p_paired),
                    "sig_wilcoxon": significance_label(p_wilcox),
                })
    return pd.DataFrame(rows)


def hybrid_paired_tests(df_hybrid: pd.DataFrame) -> pd.DataFrame:
    chf_strategy = "chf_struct_heavy"

    rows = []
    for emb in df_hybrid["embedder"].unique():
        sub = df_hybrid[df_hybrid["embedder"] == emb]
        chf = sub[sub["strategy"] == chf_strategy].sort_values("question_number")
        if not len(chf):
            continue


        comparisons = [s for s in sub["strategy"].unique()
                       if s != chf_strategy]

        for other in comparisons:
            other_df = sub[sub["strategy"] == other].sort_values("question_number")
            if not len(other_df):
                continue


            metric_cols = [c for c in ["f1_k", "recall_k", "precision_k",
                                        "f1_5", "recall_5", "precision_5",
                                        "f1_10", "recall_10", "precision_10",
                                        "f1_20", "recall_20", "precision_20"]
                           if c in chf.columns and c in other_df.columns]

            for metric in metric_cols:
                merged = pd.merge(
                    chf[["question_number", metric]],
                    other_df[["question_number", metric]],
                    on="question_number", suffixes=("_chf", "_other")
                )
                if len(merged) < 10:
                    continue

                a = merged[f"{metric}_chf"].values
                b = merged[f"{metric}_other"].values
                t_stat, p_paired, n = paired_t_test(a, b)
                w_stat, p_wilcox, n_nz = wilcoxon_signed_rank(a, b)
                d_paired = cohens_d_paired(a - b)
                rows.append({
                    "embedder": emb,
                    "chf_strategy": chf_strategy,
                    "vs_strategy": other,
                    "metric": metric,
                    "n_pairs": n,
                    "mean_chf": float(a.mean()),
                    "mean_other": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "t_stat": t_stat,
                    "p_paired_t": p_paired,
                    "p_wilcoxon": p_wilcox,
                    "cohens_d": d_paired,
                    "effect_size": effect_size_label(d_paired),
                    "sig_paired_t": significance_label(p_paired),
                    "sig_wilcoxon": significance_label(p_wilcox),
                })
    return pd.DataFrame(rows)


def cross_embedder_welch(df_chf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    chf_strategy = "chf_full_w_struct_heavy"
    base_strategy = "cosine_doc_level"

    embs = list(df_chf["embedder"].unique())
    for i, emb_a in enumerate(embs):
        for emb_b in embs[i+1:]:
            for emb_label, emb_filter in [(emb_a, emb_a), (emb_b, emb_b)]:
                pass


            sub_a = df_chf[df_chf["embedder"] == emb_a]
            sub_b = df_chf[df_chf["embedder"] == emb_b]

            chf_a = sub_a[sub_a["strategy"] == chf_strategy].sort_values("question_number")
            base_a = sub_a[sub_a["strategy"] == base_strategy].sort_values("question_number")
            chf_b = sub_b[sub_b["strategy"] == chf_strategy].sort_values("question_number")
            base_b = sub_b[sub_b["strategy"] == base_strategy].sort_values("question_number")

            if not (len(chf_a) and len(base_a) and len(chf_b) and len(base_b)):
                continue

            merged_a = pd.merge(chf_a[["question_number", "f1_k"]],
                                base_a[["question_number", "f1_k"]],
                                on="question_number", suffixes=("_chf", "_base"))
            merged_b = pd.merge(chf_b[["question_number", "f1_k"]],
                                base_b[["question_number", "f1_k"]],
                                on="question_number", suffixes=("_chf", "_base"))

            gain_a = merged_a["f1_k_chf"].values - merged_a["f1_k_base"].values
            gain_b = merged_b["f1_k_chf"].values - merged_b["f1_k_base"].values

            t_stat, p = welch_t_test(gain_a, gain_b)
            d = cohens_d(gain_a, gain_b)
            rows.append({
                "embedder_a": emb_a,
                "embedder_b": emb_b,
                "metric": "f1_k_gain (chf - cosine_doc_level)",
                "n_a": len(gain_a),
                "n_b": len(gain_b),
                "mean_gain_a": float(gain_a.mean()),
                "mean_gain_b": float(gain_b.mean()),
                "welch_t": t_stat,
                "welch_p": p,
                "cohens_d": d,
                "effect_size": effect_size_label(d),
                "sig": significance_label(p),
            })
    return pd.DataFrame(rows)


def apply_holm_per_group(df: pd.DataFrame, p_col: str, group_cols: List[str]) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for _, group in df.groupby(group_cols):
        rejected = holm_bonferroni(group[p_col].tolist(), alpha=0.05)
        for idx, r in zip(group.index, rejected):
            out.loc[idx] = r
    return out




def main():
    print("=" * 78)
    print("Testes estatísticos CHF-RAG vs baselines")
    print("=" * 78)

    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("RESUMO ESTATÍSTICO — CHF-RAG vs baselines")
    summary_lines.append("=" * 78)


    if CHF_CSV.exists():
        print(f"\nCarregando {CHF_CSV} ...")
        df_chf = pd.read_csv(CHF_CSV)
        print(f"  Linhas: {len(df_chf)}  Embedders: {df_chf['embedder'].unique()}")

        print("\n[1/3] Paired t-test + Wilcoxon: CHF struct_heavy vs cada baseline")
        df_paired = chf_paired_tests(df_chf)


        df_paired["sig_paired_holm"] = apply_holm_per_group(
            df_paired, "p_paired_t", ["embedder", "metric"]
        )
        df_paired["sig_wilcox_holm"] = apply_holm_per_group(
            df_paired, "p_wilcoxon", ["embedder", "metric"]
        )

        df_paired.to_csv(OUTPUT_DIR / "paired_tests_chf.csv", index=False)
        print(f"  Salvo: paired_tests_chf.csv ({len(df_paired)} comparações)")


        f1_only = df_paired[df_paired["metric"] == "f1_k"].copy()
        summary_lines.append("\n--- 1. CHF vs Baselines (paired t-test, F1@10) ---")
        for emb in f1_only["embedder"].unique():
            summary_lines.append(f"\n[{emb}]")
            sub = f1_only[f1_only["embedder"] == emb]
            for _, row in sub.iterrows():
                vs = row["comparison"].split(" vs ")[1]
                summary_lines.append(
                    f"  vs {vs:25s} | Δ={row['delta']:+.4f} | "
                    f"t={row['t_stat']:+.3f} p={row['p_paired_t']:.2e} {row['sig_paired_t']:4s} | "
                    f"d={row['cohens_d']:+.3f} ({row['effect_size']:7s}) | "
                    f"Holm: {'reject' if row['sig_paired_holm'] else 'fail'}"
                )

        print("\n[2/3] Welch t-test cross-embedder")
        df_welch = cross_embedder_welch(df_chf)
        df_welch.to_csv(OUTPUT_DIR / "welch_cross_embedder.csv", index=False)
        print(f"  Salvo: welch_cross_embedder.csv ({len(df_welch)} comparações)")

        summary_lines.append("\n\n--- 2. Welch cross-embedder: CHF gain (vs cosine_doc_level) ---")
        summary_lines.append("Pergunta: o ganho do CHF é o mesmo entre embedders?")
        for _, row in df_welch.iterrows():
            summary_lines.append(
                f"  {row['embedder_a']:20s} vs {row['embedder_b']:20s} | "
                f"gain_a={row['mean_gain_a']:+.4f} gain_b={row['mean_gain_b']:+.4f} | "
                f"p={row['welch_p']:.2e} {row['sig']:4s} | d={row['cohens_d']:+.3f}"
            )
    else:
        print(f"AVISO: {CHF_CSV} não encontrado, pulando paired tests")


    if HYBRID_CSV.exists():
        print(f"\nCarregando {HYBRID_CSV} ...")
        df_hyb = pd.read_csv(HYBRID_CSV)
        print(f"  Linhas: {len(df_hyb)}  Embedders: {df_hyb['embedder'].unique()}")

        print("\n[3/3] Paired t-test: CHF puro vs cada pipeline (BM25, cosine, +rerank, ...)")
        df_hyb_paired = hybrid_paired_tests(df_hyb)
        df_hyb_paired["sig_paired_holm"] = apply_holm_per_group(
            df_hyb_paired, "p_paired_t", ["embedder", "metric"]
        )
        df_hyb_paired.to_csv(OUTPUT_DIR / "paired_tests_hybrid.csv", index=False)
        print(f"  Salvo: paired_tests_hybrid.csv ({len(df_hyb_paired)} comparações)")


        f1_hyb = df_hyb_paired[df_hyb_paired["metric"] == "f1_k"].copy()
        summary_lines.append("\n\n--- 3. CHF puro vs pipelines (paired t-test, F1@10) ---")
        for emb in f1_hyb["embedder"].unique():
            summary_lines.append(f"\n[{emb}]")
            sub = f1_hyb[f1_hyb["embedder"] == emb]
            for _, row in sub.iterrows():
                summary_lines.append(
                    f"  vs {row['vs_strategy']:40s} | Δ={row['delta']:+.4f} | "
                    f"p={row['p_paired_t']:.2e} {row['sig_paired_t']:4s} | "
                    f"d={row['cohens_d']:+.3f} ({row['effect_size']:7s}) | "
                    f"Holm: {'reject' if row['sig_paired_holm'] else 'fail'}"
                )


        rec_hyb = df_hyb_paired[df_hyb_paired["metric"] == "recall_k"].copy()
        for emb in rec_hyb["embedder"].unique():
            summary_lines.append(f"\n[{emb}]")
            sub = rec_hyb[rec_hyb["embedder"] == emb]
            for _, row in sub.iterrows():
                summary_lines.append(
                    f"  vs {row['vs_strategy']:40s} | Δ={row['delta']:+.4f} | "
                    f"p={row['p_paired_t']:.2e} {row['sig_paired_t']:4s} | "
                    f"d={row['cohens_d']:+.3f} ({row['effect_size']:7s})"
                )
    else:
        print(f"AVISO: {HYBRID_CSV} não encontrado, pulando hybrid tests")


    summary = "\n".join(summary_lines)
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\nOutputs em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
