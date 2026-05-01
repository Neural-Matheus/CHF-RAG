

This guide maps every table in the paper to the script that produces it,
the output file it writes, and the cells in the paper that come from that
file. Run the stages in order; every later stage consumes an output of an
earlier one.

All commands assume the working directory is the repository root and the
container image has been built (`docker compose build` or `docker build -t
chf-rag .`). All output paths are inside the mounted `./outputs/` volume.



```bash
docker compose up chf-rag
```

This stage embeds the BR-TaxQA-R corpus with each of the four bi-encoders,
builds the three Laplacians, diagonalises the coupled supra-Laplacian, runs
the heat-flow over a logarithmic grid of 31 time points per query, and
records per-query metrics for every retrieval strategy (Cosine Doc-Level
baseline, CHF-RAG with Struct Heavy weights, and the three channel-only
ablations).

Outputs:

- `outputs/coupled_heatflow/chf_raw.csv` — per-query Precision/Recall/F1
  at top-10, plus `t*` and the three CWR values, for every (embedder,
  strategy) pair.
- `outputs/coupled_heatflow/chf_summary.csv` — strategy means, used as the
  basis of Table 1.
- `outputs/coupled_heatflow/operator_diagnostics.csv` — λ_min, spectral
  gap, and other per-embedder operator properties.

Tables produced:

| Table | Cells | Source column |
|-------|-------|---------------|
| Table 1 (retrieval performance) | 24 | `chf_raw.csv` → mean over `precision_k`, `recall_k`, `f1_k` per (embedder, strategy) |
| Table 3 (channel ablation) | 12 | `chf_raw.csv` rows where `strategy ∈ {sem_only, cit_only, hier_only}` |

Expected runtime: roughly 90 to 120 minutes on an A6000, dominated by the
four bi-encoder embedding passes over 34,753 chunks.



```bash
docker compose up hybrid-rerank
```

Runs the ten reranking pipelines (five candidate generators × two BGE
rerankers) on E5 and MiniLM, plus the two non-reranking baselines. Reranks
the top-100 candidates from each retriever down to the top-10 evaluated
list.

Outputs:

- `outputs/hybrid_rerank/hybrid_summary.csv` — F1@10 and Recall@10 per
  pipeline per embedder.
- `outputs/hybrid_rerank/hybrid_raw.csv` — per-query metrics (used by
  Stage 3 for the paired t-tests in Table 8).

Tables produced:

| Table | Cells | Source |
|-------|-------|--------|
| Table 7 (representative pipelines) | 26 | `hybrid_summary.csv` (E5 and MiniLM rows for the 7 listed pipelines) |

Expected runtime: roughly 90 to 120 minutes on an A6000 (dominated by
85,500 GPU forward passes through `bge-reranker-v2-m3` per embedder).



```bash
docker compose up stats
```

Computes paired t-tests of CHF-RAG against every other retrieval strategy
on per-query F1@10 and Recall@10, applies Holm-Bonferroni correction over
the 22 comparisons against the reranking pipelines, and runs a
cross-embedder Welch test on per-query gains.

Outputs:

- `outputs/statistical_tests/paired_tests.csv` — Stage 1 ablations and
  CHF-vs-cosine comparisons (drives Table 2 and the channel-ablation
  significance markers in Table 3).
- `outputs/statistical_tests/paired_tests_hybrid.csv` — CHF-vs-rerankers
  comparisons (drives Table 8).
- `outputs/statistical_tests/welch_tests.csv` — cross-embedder gains
  (drives the inline Welch result reported in Section 5.3).

Tables produced:

| Table | Cells | Source |
|-------|-------|--------|
| Table 2 (paired t-tests vs Cosine) | 16 | `paired_tests.csv` rows with `strategy_b == cosine_doc_level` |
| Table 3 (ablation significance markers) | embedded | `paired_tests.csv` rows with `strategy_b ∈ {sem_only, cit_only, hier_only}` |
| Table 8 (CHF vs reranking pipelines) | 40 | `paired_tests_hybrid.csv` |

Expected runtime: under one minute (CPU only).



```bash
docker compose up xai
```

Aggregates `t*` distributions, CWR per channel, and Spearman correlations
between `t*` and `ΔF1` from the per-query data already produced by Stage 1.

Outputs:

- `outputs/xai_analysis/t_star_quantiles.csv` — the percentile rows of
  Table 5.
- `outputs/xai_analysis/cwr_channels.csv` — Table 6.
- `outputs/xai_analysis/spearman_correlations.csv` — the ρ values reported
  in the conclusion (RQ3).

Tables produced:

| Table | Cells | Source |
|-------|-------|--------|
| Table 5 (`t*` distribution) | 20 | `t_star_quantiles.csv` |
| Table 6 (CWR per channel) | 16 | `cwr_channels.csv` |

Expected runtime: under thirty seconds (CPU only, reads Stage 1 outputs).



```bash
docker compose up reanalyze
```

Re-annotates each evaluation query with its number of explicit
ground-truth references and partitions the per-query F1@10 values into the
four strata reported in Table 4.

Outputs:

- `outputs/reanalyze/stratified_f1.csv` — Table 4.
- `outputs/reanalyze/strata_counts.csv` — the n=3, 29, 34, 28 counts.

Tables produced:

| Table | Cells | Source |
|-------|-------|--------|
| Table 4 (F1@10 by stratum) | 12 | `stratified_f1.csv` |

Expected runtime: under thirty seconds (CPU only).



```bash
docker compose up --build chf-rag hybrid-rerank stats xai reanalyze
```

The dependencies declared in `docker-compose.yml` make stages 2–5 wait for
their prerequisites to finish, so this single command reproduces all eight
tables of the paper end to end.



Several columns are computed and persisted in the per-query CSVs but do not
appear in any table or figure of the paper. They are intermediate XAI
diagnostics that we explored during development; we kept them in the output
for transparency and so that reviewers can inspect them, but the published
analysis ultimately settled on the Coupling Work Ratio (CWR) and the
distribution of `t*` as the primary structural-attribution signals.

The auxiliary columns are:

- `provenance_entropy` (in `chf_raw.csv`) — Shannon entropy of the
  channel-of-origin distribution among the top-k retrieved documents,
  produced by `flow_signatures_for_topk` followed by `provenance_entropy`
  in `src/heatflow_metrics.py`. This is an alternative XAI signal that
  partially overlaps with CWR; we found CWR more interpretable and
  reported it instead.
- `erc_star` and `auc_f1` (in `chf_raw.csv`) — exponential-decay-weighted
  effective relevance and the area under the F1(t) curve, both computed
  from the same per-query F1@k(t) sweep that selects `t*`. These are
  alternative summaries of the diffusion trajectory; the paper uses `t*`
  itself.
- `n_diploma_mentions` (in `chf_raw.csv`) — number of explicit legal
  diploma mentions in each query, produced by
  `count_diploma_mentions`. Used internally to debug the
  direct/conceptual classifier; the paper reports the
  `classify_question_by_ref_count` stratification (Table 4) instead.
- Coupling-lift values logged at INFO level by `coupling_lift` —
  written to the run log only, never persisted to a CSV or referenced in
  the paper.

Two top-level imports in `scripts/run_coupled_heatflow.py` —
`cheeger_bound_check` and `aggregate_t_star` — are residual from earlier
development iterations and are not invoked anywhere in the script. They
are harmless: the module loads, but the corresponding diagnostics are
not computed.

None of the above affects any cell, plot, or statistical claim in the
paper. The values that appear in Tables 1–8 come exclusively from
`f1_k`, `precision_k`, `recall_k`, `t_star`, `cwr_*`, and the
paired/Welch test outputs in `outputs/statistical_tests/`.



The numerical pipeline is deterministic given a fixed corpus and fixed
model checkpoints:

- All bi-encoder weights are pinned in `src/embedder.py` (E5,
  Legal-BERTimbau, STJIRIS-Legal, multilingual MiniLM L12 v2).
- The BGE rerankers use `BAAI/bge-reranker-v2-m3` and
  `BAAI/bge-reranker-base`, also pinned.
- BM25, the supra-Laplacian construction, and the heat-flow grid are
  deterministic.

Tiny floating-point differences (1e-4 and below) may arise across
GPU/driver versions, but the cells reported in the paper round to four
decimal places and do not depend on these. We have verified the run on
PyTorch 2.4 / CUDA 12.4 against PyTorch 2.4 / CUDA 12.6.



The `tests/` folder contains sanity checks that do not require the full
corpus and run in under a minute:

```bash
docker compose run chf-rag python -m pytest tests/
```

These verify the supra-Laplacian construction, the energy decay bound, and
the small-graph behaviour of the CWR.
