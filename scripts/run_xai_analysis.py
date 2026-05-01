import sys
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("/app/outputs/xai_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHF_CSV = Path("/app/outputs/coupled_heatflow/chf_raw.csv")


def main():
    if not CHF_CSV.exists():
        print(f"ERRO: {CHF_CSV} não existe")
        sys.exit(1)

    df = pd.read_csv(CHF_CSV)
    print(f"Loaded {len(df)} rows. Strategies: {df['strategy'].nunique()}, Embedders: {df['embedder'].nunique()}")


    chf = df[df["strategy"] == "chf_full_w_struct_heavy"].copy()
    print(f"CHF struct_heavy rows: {len(chf)}")

    out_lines = []
    out_lines.append("=" * 78)
    out_lines.append("XAI ANALYSIS — CHF-RAG t*, CWR, provenance entropy")
    out_lines.append("=" * 78)


    out_lines.append("\n[1] DISTRIBUIÇÃO DE t* POR EMBEDDER")
    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'n':>4s} {'median':>8s} {'Q1':>8s} {'Q3':>8s} {'max':>8s} {'%t*=0':>8s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb]
        ts = sub["t_star"].dropna().values
        if len(ts) == 0:
            continue
        med = np.median(ts)
        q1, q3 = np.percentile(ts, [25, 75])
        mx = np.max(ts)
        pct_zero = (ts < 0.001).mean() * 100
        out_lines.append(f"{emb:<25s} {len(ts):>4d} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} {mx:>8.3f} {pct_zero:>7.1f}%")


    out_lines.append("\n[2] COUPLING WORK RATIO — média per query (β = 0.10)")
    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'CWR_sem':>10s} {'CWR_cit':>10s} {'CWR_hier':>10s} {'CWR_mean':>10s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb]
        s_sem = sub["cwr_sem"].dropna().mean()
        s_cit = sub["cwr_cit"].dropna().mean()
        s_hier = sub["cwr_hier"].dropna().mean()
        s_mean = sub["cwr_mean"].dropna().mean()
        out_lines.append(f"{emb:<25s} {s_sem:>10.4f} {s_cit:>10.4f} {s_hier:>10.4f} {s_mean:>10.4f}")


    out_lines.append("\n[3] PER-QUERY DOMINANT CHANNEL — fração de queries onde cada canal lidera")
    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'sem domina':>12s} {'cit domina':>12s} {'hier domina':>12s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb]
        valid = sub.dropna(subset=["cwr_sem", "cwr_cit", "cwr_hier"])
        if len(valid) == 0:
            continue
        sem_wins = (valid["cwr_sem"] >= valid[["cwr_cit", "cwr_hier"]].max(axis=1)).mean() * 100
        cit_wins = (valid["cwr_cit"] >= valid[["cwr_sem", "cwr_hier"]].max(axis=1)).mean() * 100
        hier_wins = (valid["cwr_hier"] >= valid[["cwr_sem", "cwr_cit"]].max(axis=1)).mean() * 100
        out_lines.append(f"{emb:<25s} {sem_wins:>11.1f}% {cit_wins:>11.1f}% {hier_wins:>11.1f}%")


    out_lines.append("\n[4] PROVENANCE ENTROPY — distribuição da diversidade de fontes")
    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'mean':>8s} {'median':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb]
        H = sub["provenance_entropy"].dropna().values
        if len(H) == 0:
            continue
        out_lines.append(f"{emb:<25s} {H.mean():>8.4f} {np.median(H):>8.4f} {H.std():>8.4f} {H.min():>8.4f} {H.max():>8.4f}")


    out_lines.append("\n[5] CORRELAÇÃO t* vs ΔF1 (CHF − cosine baseline at t=0)")
    out_lines.append("    Hipótese: queries que precisam mais difusão também ganham mais")
    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'Pearson r':>10s} {'Spearman ρ':>12s} {'n':>5s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb].dropna(subset=["t_star", "delta_f1"])
        if len(sub) < 10:
            continue
        ts = sub["t_star"].values
        df_vals = sub["delta_f1"].values

        if ts.std() > 0 and df_vals.std() > 0:
            r_p = np.corrcoef(ts, df_vals)[0, 1]
        else:
            r_p = float("nan")

        ranks_t = pd.Series(ts).rank().values
        ranks_f = pd.Series(df_vals).rank().values
        if ranks_t.std() > 0 and ranks_f.std() > 0:
            r_s = np.corrcoef(ranks_t, ranks_f)[0, 1]
        else:
            r_s = float("nan")
        out_lines.append(f"{emb:<25s} {r_p:>+10.4f} {r_s:>+12.4f} {len(sub):>5d}")

    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'%t*=0':>8s} {'%t*∈(0,0.5]':>12s} {'%t*∈(0.5,2]':>12s} {'%t*>2':>8s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb]
        ts = sub["t_star"].dropna().values
        if len(ts) == 0:
            continue
        z = (ts < 0.001).mean() * 100
        s = ((ts >= 0.001) & (ts <= 0.5)).mean() * 100
        m = ((ts > 0.5) & (ts <= 2.0)).mean() * 100
        l = (ts > 2.0).mean() * 100
        out_lines.append(f"{emb:<25s} {z:>7.1f}% {s:>11.1f}% {m:>11.1f}% {l:>7.1f}%")


    out_lines.append("-" * 78)
    out_lines.append(f"{'Embedder':<25s} {'t*=0':>10s} {'t*>0':>10s} {'diff':>10s}")
    for emb in sorted(chf["embedder"].unique()):
        sub = chf[chf["embedder"] == emb].dropna(subset=["t_star", "delta_f1"])
        if len(sub) < 10:
            continue
        zero_mask = sub["t_star"] < 0.001
        df_zero = sub[zero_mask]["delta_f1"].mean() if zero_mask.any() else float("nan")
        df_pos = sub[~zero_mask]["delta_f1"].mean() if (~zero_mask).any() else float("nan")
        out_lines.append(f"{emb:<25s} {df_zero:>+10.4f} {df_pos:>+10.4f} {df_pos-df_zero:>+10.4f}")

    summary = "\n".join(out_lines)
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write(summary)
    print(summary)
    print(f"\nSalvo em {OUTPUT_DIR}/summary.txt")


if __name__ == "__main__":
    main()
