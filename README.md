

Reference implementation accompanying the paper *Coupled Heat-Flow RAG: A
Multiplex Diffusion with Auditable Thermodynamic Time for Brazilian
Tax Law Retrieval*.

CHF-RAG models document ranking as a continuous heat-diffusion process across
a multiplex graph of three relevance channels: dense semantic similarity,
historical co-citation, and Kelsenian normative hierarchy. The optimal
diffusion time `t*` is exposed as a per-query Explainable AI signal.


This repository contains exactly the code that produced the numbers in the
paper. There is no auxiliary or exploratory material.

- `src/` — eight modules implementing the supra-Laplacian operator, the
  coupled heat-flow PDE, the Coupling Work Ratio, the bi-encoder/BM25/BGE
  baselines, and the BR-TaxQA-R loader.
- `scripts/` — five executable scripts, one per pipeline stage. Each maps
  directly to one or more tables in the paper (see [REPRODUCE.md](REPRODUCE.md)).
- `tests/` — sanity checks for the diffusion math (PSD operator, energy
  decay bound, eigendecomposition consistency) and small dry-runs of the
  retrieval pipelines.
- `Dockerfile` and `docker-compose.yml` — pinned environment matching the
  evaluation server (PyTorch 2.4 / CUDA 12.4, single NVIDIA RTX A6000 48 GB).



```bash
git clone https://github.com/<user>/chf-rag.git
cd chf-rag


docker compose up --build chf-rag hybrid-rerank stats xai reanalyze


docker compose up chf-rag
```

Outputs land in `./outputs/`. See [REPRODUCE.md](REPRODUCE.md) for the
mapping between scripts, output files, and paper tables.


Given a corpus of N legal documents:

1. Build three undirected weighted graphs over the same N nodes:
   `G_sem` (semantic, k-NN with k=8 over chunk-pair cosine), `G_cit`
   (co-citation Jaccard), `G_hier` (Kelsenian level proximity).
2. Construct the supra-Laplacian operator `M` of size 3N × 3N, with each
   block-diagonal entry being a normalized Laplacian and the off-diagonal
   blocks being uniform coupling β·I (β = 0.10).
3. Apply Tikhonov regularization to obtain `M_ε = M + εI` (ε = 10⁻³) and
   diagonalize once at indexing time.
4. For each query, initialize heat with a temperature-scaled softmax over
   query-document cosine, run the diffusion `U(t) = exp(-t M_ε) U_0` over a
   logarithmic grid of 31 time points, and rank documents by a weighted
   combination of the three channels.
5. The diffusion time `t* = argmax_t F1@k(t)` is reported per query as the
   XAI signal; the per-channel Coupling Work Ratio decomposes it by
   structural source.

The full mathematical derivation, including the energy bound
`E(t) ≤ E(0) exp(-2λ_min t)`, is in Section 3 of the paper.


The reported numbers were produced on a single workstation with one NVIDIA
RTX A6000 (48 GB VRAM), CUDA 12.6, and 64 GB of system RAM. The bi-encoders
and BGE rerankers run on the GPU; the supra-Laplacian diagonalization and
heat evolution run on CPU (the operator is 1434 × 1434, well within reach).

End-to-end runtime, including embedding the corpus with all four bi-encoders
and the full reranking sweep, is approximately three to four hours on this
hardware.


The experiments use BR-TaxQA-R [Domingos Júnior et al., BRACIS 2025], a
Brazilian personal-income-tax question-answering dataset with 715 queries,
478 legal documents, and human-annotated ground-truth references. The
loader in `src/data_loader.py` downloads the public release on first use and
caches it under `.cache/`.

We use the first 200 questions of the Q&A split. Of those, 171 carry
ground-truth reference annotations and form the evaluation set.


| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.4 (CUDA 12.4) |
| sentence-transformers | ≥ 3.1 |
| transformers | ≥ 4.45 |
| numpy / scipy / pandas | latest 2.x / 1.13+ / 2.2+ |

Exact pins are in `requirements.txt`. The Docker image installs everything
from those pins, so the artefact is self-contained.

MIT. See [LICENSE](LICENSE).
