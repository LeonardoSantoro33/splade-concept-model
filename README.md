# SPLADE Concept Model

**Compressing SPLADE into k dense "concepts" with a learned (or random) linear projection, and what that does to in-domain and out-of-domain retrieval.**

Research code and results for a project carried out at Sapienza University of Rome, using the computing
facilities of ISTI-CNR. Everything in this repository was run with exact (non-approximate) search over the
full MS MARCO passage corpus (8.84M passages).

---

## Overview

SPLADE maps a text to a sparse 30,522-dimensional vector over the BERT vocabulary. The **Concept Model**
projects that vector into a dense space of k "concepts":

$$g(x) = W \cdot \mathrm{SPLADE}(x), \qquad W \in \mathbb{R}^{k \times 30522}, \qquad \mathrm{score}(q,d) = g(q)\cdot g(d)$$

W (and optionally the SPLADE encoder itself) is trained with an InfoNCE loss on MS MARCO, with in-batch negatives
plus 7 hard negatives per query mined with SPLADE. The base model is `naver/splade-cocondenser-ensembledistil`
(we verified that it is the model that produced the pre-encoded vectors we were given: mean cosine 1.0000).

Variants studied, for k = 256, 512, 1024, 2048, 4096, 8192:

| Name | W | SPLADE encoder (BERT) |
|---|---|---|
| **Random W** | random, never trained | original, fixed |
| **Frozen** | trained | original, fixed |
| **e2e** | trained (from random) | re-trained |
| **e2e with fixed W** (k = 1024 only) | random, fixed | re-trained |
| **Hybrids** | any of the above, score mixed with SPLADE: `(1-λ)·SPLADE·SPLADE + λ·g·g` | |

Evaluation: MS MARCO dev (6,980 queries, MRR@10), TREC DL 2019 / 2020 (nDCG@10), and zero-shot BEIR
(NFCorpus, SciFact, FiQA; nDCG@10, mean of the three). Hyper-parameters and λ are chosen **only** on 1,000
held-out training queries, never on dev or BEIR. Differences are tested per query with a paired bootstrap
and permutation test (`12_significance.py`).

## Main results

Percentages are relative to SPLADE (MS MARCO dev MRR@10 = 0.3828, BEIR mean nDCG@10 = 0.4662).
Full tables (DL19/DL20, recall, hybrids, significance tests) are in [`docs/RESULTS.md`](docs/RESULTS.md).

**Concept Model alone**

| k | bytes/doc (fp16) | Random W dev | Frozen dev | e2e dev | Random W BEIR | Frozen BEIR | e2e BEIR |
|---|---|---|---|---|---|---|---|
| 256 | 512 | 0.2544 (66.5%) | 0.2789 (72.9%) | 0.3424 (89.4%) | 0.3117 (66.9%) | 0.3356 (72.0%) | 0.3372 (72.3%) |
| 512 | 1,024 | 0.3129 (81.7%) | 0.3256 (85.1%) | 0.3598 (94.0%) | 0.3929 (84.3%) | 0.4016 (86.1%) | 0.4027 (86.4%) |
| 1024 | 2,048 | 0.3454 (90.2%) | 0.3451 (90.2%) | 0.3731 (97.5%) | 0.4300 (92.2%) | 0.4294 (92.1%) | 0.4291 (92.0%) |
| 2048 | 4,096 | 0.3634 (94.9%) | 0.3683 (96.2%) | 0.3799 (99.2%) | 0.4370 (93.7%) | 0.4431 (95.0%) | 0.4347 (93.3%) |
| 4096 | 8,192 | 0.3691 (96.4%) | 0.3717 (97.1%) | **0.3829 (100.0%)** | 0.4601 (98.7%) | 0.4598 (98.6%) | 0.4580 (98.2%) |
| 8192 | 16,384 | 0.3761 (98.2%) | 0.3766 (98.4%) | 0.3722 (97.2%) | 0.4600 (98.7%) | 0.4605 (98.8%) | 0.4541 (97.4%) |

**Mixing with SPLADE** (two sparse indexes; λ chosen on validation)

| System | MS MARCO dev MRR@10 | BEIR mean nDCG@10 | Index size |
|---|---|---|---|
| SPLADE | 0.3828 | 0.4662 | 8.5 GB |
| SPLADE + re-trained encoder, no W (`hybnoW`), k = 256, λ = 0.40 | **0.3920** (+0.0092, +2.4%, p = 0.0001) | 0.4676 (100.3%) | 15.8 GB |
| SPLADE + e2e Concept Model (`hyb`), k = 256, λ = 0.20 | 0.3901 (+0.0073, p = 0.0001) | 0.4646 (99.7%) | 15.8 GB |
| SPLADE + e2e Concept Model (`hyb`), k = 4096, λ = 0.30 | 0.3880 (+0.0052, p = 0.0001) | 0.4661 (100.0%) | 19.1 GB |

### Take-aways

1. **A linear projection of SPLADE to k dimensions, with the encoder frozen, keeps 73–98% of SPLADE's quality on MS MARCO and 72–99% on BEIR.** The percentage is about the same in and out of domain, so compressing     SPLADE does not make it worse on new datasets.
2. **k is the main lever. Learning W helps only at small k** (+0.025 MRR@10 at k = 256, +0.013 at 512). From
   k = 1024 on, a random projection is statistically indistinguishable from a learned one.
3. **Re-training the encoder (e2e) closes most of the gap with SPLADE in domain**: the e2e model reaches 97.5% of SPLADE at k = 1024 (0.3731 vs 0.3828) and ties with it at k = 4096 (0.3829 vs 0.3828), while the frozen model reaches only 90.2% and 97.1% at the same k. For the same quality, e2e needs about 4× less space than frozen (e2e at k = 1024 ≈ frozen at k = 4096, 0.3731 vs 0.3717). The gain comes almost entirely from re-training BERT, not from W: with W fixed at its random initialisation, k = 1024 gives 0.3714, against 0.3731 with trained W.
4. **The e2e gain is specific to MS MARCO.** Out of domain the e2e model does not meaningfully beat the frozen one at any k (at most +0.002 at k ≤ 512, negative from k = 2048).
   At k = 8192 it is worse than the frozen model even in domain (0.3722 vs 0.3766), although validation had predicted
   the opposite. The optimal BERT learning rate keeps falling with k (1e-5 at k = 256, 1e-7 at k = 8192).
5. **Mixing the re-trained encoder with SPLADE beats SPLADE in-domain and stays at the level of SPLADE out of domain**
   (best: `hybnoW`, k = 256). The price is a second sparse index (+86% space). Per-dataset BEIR tests against
   SPLADE show no significant difference after multiple-test correction.
6. **W is not needed at search time**: the hybrid without W is never worse than the one with W (and significantly
   better at k = 512, 1024, 8192). The Concept Model acts as a *training objective* that improves the encoder.
7. **No saving in space.** SPLADE stores about 120 non-zeros per document (~1 KB); a dense k-dimensional fp16
   vector is 2k bytes, so no variant is smaller than SPLADE at equal quality.
8. The learned "concepts" are **not interpretable** (the top tokens of a concept are incoherent and zeroing the strongest
   concept of a query does not change the rank of the relevant passage): information is distributed.

Limits: one seed per configuration; hyper-parameter recipes are not identical across k (see `docs/RESULTS.md`);
e2e queries are truncated to 64 tokens (256 for SPLADE and frozen); the 8192 results come from the same pipeline but
validation was less reliable there. The headline `hybnoW` k = 256 is the best of the k values on dev (on validation k = 256 and 512 are tied, 0.3769 vs 0.3770), so it is a best-of-k pick; every k of `hybnoW` is in `docs/RESULTS.md`.

## Repository structure

Scripts are flat and numbered in execution order, and every command in the logs refers to them by this name,
so they are deliberately not moved into a package.

```
.
├── 00_smoke_test.py                # unit tests + full pipeline on synthetic data (CPU, ~5 min)
├── 01_prepare_data.py              # read the provided Seismic .bin files, map token ids to the BERT vocab, cache as CSR
├── 02_download_msmarco.py          # MS MARCO queries / qrels / passages (+ TREC DL 2019/2020)
├── 03_check_encoder.py             # which SPLADE checkpoint reproduces the provided vectors
├── 04_baseline_splade.py           # SPLADE baseline, exact search, check vs. the provided ground truth
├── 05_encode_train_queries.py      # SPLADE vectors of the MS MARCO training queries
├── 06_train_concept_model.py       # training: --mode frozen | e2e, InfoNCE, hard negatives
├── 07_mine_negatives.py            # hard-negative mining (SPLADE or a trained model)
├── 08_eval_concept_model.py        # official evaluation (dense search), concept analysis, interventions
├── 08b_eval_streaming.py           # same, streaming exact search for frozen models with k >= 2048 (does not fit in 40 GB)
├── 08c_eval_e2e_streaming.py       # same for e2e models; re-encodes the corpus and saves the sparse shards used by 11
├── 09_lce_splade.py                # legacy: Latent Concept Expansion baseline from the earlier SPLARE project
├── 10_eval_beir.py                 # zero-shot BEIR (--splade, --ckpts, --ft-splade, --hybrid)
├── 11_eval_hybrid.py               # hybrids: λ chosen on validation, then dev / DL19 / DL20 (--variant two|single|now)
├── 12_significance.py              # paired per-query tests: bootstrap CI + t-test + permutation test
│
├── config.py                       # paths (env vars FM_DIR, WORK_DIR)
├── sparse_io.py  retrieval.py      # CSR / Seismic I/O; exact GPU search
├── splade_encoder.py               # SPLADE encoder (memory-efficient pooling)
├── concept_model.py                # W, InfoNCE, false-negative mask, checkpoints
├── metrics.py  msmarco.py  pipeline.py  utils.py
├── lce.py                          # used by the smoke test and by 09 (LCE of the earlier project)
│
├── jobs/                           # generic Slurm job scripts (gpu.sbatch, cpu.sbatch)
├── docs/GUIDE.md                   # step-by-step guide: how every result was produced
├── docs/RESULTS.md                 # all result tables, final checkpoints and hyper-parameters
├── results/ logs/ training_logs/   # not tracked: download them from the v1.0 release (see Reproducibility)
├── plots/                          # concept_model_plots.py and the figures (plots/figures/)
├── requirements.txt                # direct dependencies
└── requirements_env.txt            # exact versions of the environment used for all experiments
```

`results/`, `logs/` and `training_logs/` are not in the repository: they are archived in the [v1.0 release](https://github.com/LeonardoSantoro33/splade-concept-model/releases/tag/v1.0).

The scripts themselves write under `$WORK_DIR`. Checkpoints, embeddings and caches (about 260 GB) are **not** distributed.

## Quickstart

```bash
conda create -n splade python=3.11 && conda activate splade
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8 build used on the cluster
pip install -r requirements.txt            # or: pip install -r requirements_env.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -u 00_smoke_test.py                 # should end with: ALL TESTS PASSED (9)
```

The pipeline needs the pre-encoded MS MARCO SPLADE vectors in Seismic format (`documents.bin`, `queries.bin`, ...),
which were provided to us and are **not public**, so the full pipeline cannot be re-run from scratch without them
(you can regenerate them by encoding MS MARCO with `naver/splade-cocondenser-ensembledistil`). Paths are read from
environment variables:

```bash
export FM_DIR=/path/to/seismic/encoded/msmarco      # read-only input
export WORK_DIR=/path/to/large/scratch/splade_concept   # caches, checkpoints, results (several hundred GB for all experiments)
```

Every long step is a Slurm job; the complete sequence, with the exact commands, is in [`docs/GUIDE.md`](docs/GUIDE.md).
The figures are produced with `python plots/concept_model_plots.py` (numbers are written inside the script).

## Reproducibility

All raw training logs (115 runs) and evaluation outputs are archived as
`splade-concept-model-logs-results.tar.gz` in the
[v1.0 release](https://github.com/LeonardoSantoro33/splade-concept-model/releases/tag/v1.0).
They are not part of the repository, to keep the clone lightweight.

To restore
`logs/`, `training_logs/` and `results/`, download the archive and extract it
in the repository root:

```
curl -L -O https://github.com/LeonardoSantoro33/splade-concept-model/releases/download/v1.0/splade-concept-model-logs-results.tar.gz
tar xzf splade-concept-model-logs-results.tar.gz
```

Final numbers are summarised in [`docs/RESULTS.md`](docs/RESULTS.md); the
archive is only needed to inspect or verify individual runs.

## Legacy note on LCE

`09_lce_splade.py` and `lce.py` implement Latent Concept Expansion (a training-free expansion with a corpus
co-occurrence matrix), the baseline of the earlier SPLARE project. It is not part of the results above;
`lce.py` is kept because `00_smoke_test.py` imports it.

## References

- Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*, SIGIR 2021.
- Formal et al., *From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective* (SPLADE++), SIGIR 2022.
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS 2021.
- Bajaj et al., *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*, 2016.
- Koh et al., *Concept Bottleneck Models*, ICML 2020.
- Xiong et al., *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (ANCE), ICLR 2021.
- Bruch et al., *Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations* (Seismic), SIGIR 2024.

## License

MIT, see [LICENSE](LICENSE).

## Author

Leonardo Santoro, Sapienza University of Rome.
