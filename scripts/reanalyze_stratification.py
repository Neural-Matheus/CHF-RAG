import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_questions
from src.question_typing import (
    classify_question_by_refs,
    classify_question_by_ref_count,
    stratify_t_star,
)


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:

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


def reannotate_with_question_types(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRe-anotando perguntas a partir do dataset...")
    questions = load_questions()
    q_by_num = {str(q.number): q for q in questions}

    types_binary = []
    types_granular = []
    n_refs = []
    for qn in df["question_number"].astype(str):
        q = q_by_num.get(qn)
        if q is None:
            types_binary.append("unknown")
            types_granular.append("unknown")
            n_refs.append(0)
        else:
            types_binary.append(
                classify_question_by_refs(q.references_explicit, q.references_implicit))
            types_granular.append(
                classify_question_by_ref_count(q.references_explicit, q.references_implicit))
            n_refs.append(len(q.references_explicit) if q.references_explicit else 0)

    df["question_type"] = types_binary
    df["question_type_granular"] = types_granular
    df["n_refs_explicit"] = n_refs
    return df


def stratified_analysis(
    df: pd.DataFrame,
    type_col: str = "question_type_granular",
    embedder: str = None,
    strategies: List[str] = None,
):

    if embedder:
        df = df[df["embedder"] == embedder]
    if strategies:
        df = df[df["strategy"].isin(strategies)]

    rows = []
    type_groups = df[type_col].unique()
    type_groups = sorted([t for t in type_groups if t != "unknown"])
    print(f"\nEstratificação por '{type_col}' — grupos: {type_groups}")

    for emb in sorted(df["embedder"].unique()):
        for strat in sorted(df["strategy"].unique()):
            sub = df[(df["embedder"] == emb) & (df["strategy"] == strat)]
            if len(sub) < 5:
                continue

            row = {"embedder": emb, "strategy": strat, "n_total": len(sub)}

            t_by_group = {}
            f1_by_group = {}
            n_by_group = {}
            for grp in type_groups:
                grp_sub = sub[sub[type_col] == grp]
                if len(grp_sub) > 0:
                    t_by_group[grp] = grp_sub["t_star"].values
                    f1_by_group[grp] = grp_sub["f1_k"].values
                    n_by_group[grp] = len(grp_sub)
                    row[f"n_{grp}"] = len(grp_sub)
                    row[f"t_star_median_{grp}"] = float(np.median(grp_sub["t_star"]))
                    row[f"t_star_mean_{grp}"] = float(np.mean(grp_sub["t_star"]))
                    row[f"f1_mean_{grp}"] = float(np.mean(grp_sub["f1_k"]))


            if len(type_groups) >= 2:
                g_low, g_high = type_groups[0], type_groups[-1]
                if g_low in t_by_group and g_high in t_by_group:
                    p_t = _welch_p(t_by_group[g_low], t_by_group[g_high])
                    p_f1 = _welch_p(f1_by_group[g_low], f1_by_group[g_high])
                    row[f"welch_p_t_{g_low}_vs_{g_high}"] = p_t
                    row[f"welch_p_f1_{g_low}_vs_{g_high}"] = p_f1

            rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df_strat: pd.DataFrame, type_col: str):

    chf_full_mask = df_strat["strategy"].str.startswith("chf_full")
    isolated_mask = df_strat["strategy"].isin(
        ["sem_only", "cit_only", "hier_only", "sem_cit"])

    print("\n" + "=" * 78)
    print(f"ESTRATIFICAÇÃO POR {type_col} — médias por grupo")
    print("=" * 78)

    for emb in sorted(df_strat["embedder"].unique()):
        emb_sub = df_strat[df_strat["embedder"] == emb]
        print(f"\n[{emb}]")


        focus_strats = (
            ["chf_full_b0.10", "chf_full_w_cit_heavy", "chf_full_w_struct_heavy"]
            + ["sem_only", "cit_only", "hier_only", "sem_cit"]
        )
        for strat in focus_strats:
            row_match = emb_sub[emb_sub["strategy"] == strat]
            if not len(row_match):
                continue
            row = row_match.iloc[0]
            grp_cols = [c for c in row.index if c.startswith("t_star_mean_")]
            grps = sorted([c.replace("t_star_mean_", "") for c in grp_cols])
            
            line = f"  {strat:30s} | "
            for g in grps:
                n = int(row.get(f"n_{g}", 0))
                t_med = row.get(f"t_star_median_{g}", float("nan"))
                f1 = row.get(f"f1_mean_{g}", float("nan"))
                line += f"{g}(n={n}): t*={t_med:.3f} F1={f1:.3f} | "
            print(line)
            

            p_keys = [c for c in row.index if c.startswith("welch_p_")]
            for pk in p_keys:
                p = row.get(pk)
                if isinstance(p, float) and not np.isnan(p):
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    print(f"      └─ {pk} = {p:.3e}  {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/app/outputs/coupled_heatflow/chf_raw.csv",
                        help="Path to chf_raw.csv")
    parser.add_argument("--output-dir", default="/app/outputs/coupled_heatflow",
                        help="Where to save reanalysis CSVs")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():

        csv_path = Path("outputs/coupled_heatflow/chf_raw.csv")
    if not csv_path.exists():
        print(f"ERROR: chf_raw.csv não encontrado em {args.csv}")
        sys.exit(1)

    print(f"Carregando {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Linhas: {len(df)}  Colunas: {list(df.columns)}")
    print(f"Embedders: {df['embedder'].unique()}")
    print(f"Estratégias: {df['strategy'].nunique()}")


    df = reannotate_with_question_types(df)


    n_per_strat = df.groupby("strategy").size().iloc[0]
    n_per_emb = df.groupby("embedder").size().iloc[0]
    n_unique_q = df.groupby(["question_number", "embedder"]).size().shape[0]
    print(f"\nLinhas por estratégia: ~{n_per_strat}")
    print(f"Linhas por embedder: ~{n_per_emb}")
    print(f"(question, embedder) únicos: {n_unique_q}")

    print("\nDistribuição global (binária, deduplicada por (q,emb)):")
    one_strat = df[df["strategy"] == df["strategy"].iloc[0]]
    print(one_strat["question_type"].value_counts().to_dict())
    print("\nDistribuição global (granular, deduplicada por (q,emb)):")
    print(one_strat["question_type_granular"].value_counts().to_dict())


    df_binary = stratified_analysis(df, type_col="question_type")
    df_binary.to_csv(Path(args.output_dir) / "stratified_binary.csv", index=False)
    print_summary(df_binary, "question_type")


    df_granular = stratified_analysis(df, type_col="question_type_granular")
    df_granular.to_csv(Path(args.output_dir) / "stratified_granular.csv", index=False)
    print_summary(df_granular, "question_type_granular")

    print("\n" + "=" * 78)
    print(f"Salvos em: {args.output_dir}/stratified_binary.csv  e  stratified_granular.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
