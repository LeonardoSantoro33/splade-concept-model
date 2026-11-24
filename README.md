# SPLADE Concept Model

**Compressing SPLADE into k dense "concepts" with a learned (or random) linear projection, and what that does to in-domain and out-of-domain retrieval.**

Research code and results for a project carried out at Sapienza University of Rome, using the computing
facilities of ISTI-CNR. Everything in this repository was run with exact (non-approximate) search over the
full MS MARCO passage corpus (8.84M passages) and the full NQ corpus (2.68M passages).

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
(NFCorpus, SciFact, FiQA, NQ; nDCG@10, mean of the four). Hyper-parameters and λ are chosen **only** on 1,000
held-out training queries, never on dev or BEIR. Differences are tested per query with a paired bootstrap
and permutation test (`src/12_significance.py`).

## Main results

Percentages are relative to SPLADE (MS MARCO dev MRR@10 = 0.3828, BEIR mean nDCG@10 = 0.4662).
Full tables (DL19/DL20, recall, hybrids, significance tests) are in [`docs/RESULTS.md`](docs/RESULTS.md).

**Concept Model alone**

| k | bytes/doc (fp16) | Random W dev | Frozen dev | e2e dev | Random W BEIR | Frozen BEIR | e2e BEIR |
|---|---|---|---|---|---|---|---|
| 256 | 512 | 0.2544 (66.5%) | 0.2789 (72.9%) | 0.3424 (89.4%) | 0.3003 (62.0%) | 0.3351 (69.2%) | 0.3585 (74.0%) |
| 512 | 1,024 | 0.3129 (81.7%) | 0.3256 (85.1%) | 0.3598 (94.0%) | 0.3935 (81.2%) | 0.4018 (82.9%) | 0.4188 (86.4%) |
| 1024 | 2,048 | 0.3454 (90.2%) | 0.3451 (90.2%) | 0.3731 (97.5%) | 0.4380 (90.4%) | 0.4421 (91.3%) | 0.4442 (91.7%) |
| 2048 | 4,096 | 0.3634 (94.9%) | 0.3683 (96.2%) | 0.3799 (99.2%) | 0.4530 (93.5%) | 0.4576 (94.5%) | 0.4536 (93.6%) |
| 4096 | 8,192 | 0.3691 (96.4%) | 0.3717 (97.1%) | **0.3829 (100.0%)** | 0.4747 (98.0%) | 0.4740 (97.8%) | 0.4748 (98.0%) |
| 8192 | 16,384 | 0.3761 (98.2%) | 0.3766 (98.4%) | 0.3722 (97.2%) | 0.4769 (98.4%) | 0.4756 (98.2%) | 0.4704 (97.1%) |

**Mixing with SPLADE** (two sparse indexes; λ chosen on validation)

| System | MS MARCO dev MRR@10 | BEIR mean nDCG@10 | Index size |
|---|---|---|---|
| SPLADE | 0.3828 | 0.4662 | 8.5 GB |
| SPLADE + re-trained encoder, no W (`hybnoW`), k = 256, λ = 0.40 | **0.3920** (+0.0092, +2.4%, p <= 0.0001) | 0.4872 (100.6%) | 15.8 GB |
| SPLADE + e2e Concept Model (`hyb`), k = 256, λ = 0.20 | 0.3901 (+0.0073, p <= 0.0001) | 0.4838 (99.9%) | 15.8 GB |
| SPLADE + e2e Concept Model (`hyb`), k = 4096, λ = 0.30 | 0.3880 (+0.0052, p <= 0.0001) | 0.4853 (100.2%) | 19.1 GB |

### Take-aways

1. **A linear projection of SPLADE to k dimensions, with the encoder frozen, keeps 73–98% of SPLADE's quality on MS MARCO and 69–98% on BEIR.** The percentage is about the same in and out of domain from k = 1024 on (within 2 points), while at small k it is slightly lower out of domain (72.9% vs 69.2% at k = 256, 85.1% vs 82.9% at k = 512). So compressing SPLADE does not make it meaningfully worse on new datasets, except at the smallest k.
2. **k is the main lever. Learning W helps only at small k** (+0.025 MRR@10 at k = 256, +0.013 at 512). From
   k = 1024 on, a random projection is statistically indistinguishable from a learned one.
3. **Re-training the encoder (e2e) closes most of the gap with SPLADE in domain**: the e2e model reaches 97.5% of SPLADE at k = 1024 (0.3731 vs 0.3828) and ties with it at k = 4096 (0.3829 vs 0.3828), while the frozen model reaches only 90.2% and 97.1% at the same k. For the same quality, e2e needs about 4× less space than frozen (e2e at k = 1024 ≈ frozen at k = 4096, 0.3731 vs 0.3717). The gain comes almost entirely from re-training BERT, not from W: with W fixed at its random initialisation, k = 1024 gives 0.3714, against 0.3731 with trained W.
4. **The e2e gain is mostly specific to MS MARCO.** Out of domain, e2e matches the frozen model at k ≥ 1024 (and is slightly below it at 2048 and 8192, −0.004/−0.005). It stays ahead at k = 256 and 512 (+0.023 and +0.017 on the four-dataset mean), but this advantage comes almost entirely from NQ (+0.089 and +0.065); on NFCorpus, SciFact and FiQA the two are equal.
   At k = 8192 it is worse than the frozen model even in domain (0.3722 vs 0.3766), although validation had predicted
   the opposite.
5. **Mixing the re-trained encoder with SPLADE beats SPLADE in-domain and is at least at its level out of domain**
   (best: `hybnoW`, k = 256; BEIR mean 0.4872 vs 0.4844). Per-dataset tests (nDCG@10, Bonferroni over 4 datasets): on par on NFCorpus, SciFact and FiQA, significantly better on NQ (+0.0070, p = 0.0004). The price is a second sparse index (+86% space).
6. **No saving in space.** SPLADE stores about 120 non-zeros per document (~1 KB). A dense k-dimensional fp16
   vector is 2k bytes, so no variant is smaller than SPLADE at equal quality.

## Repository structure

The only script outside `src/` is `plots/concept_model_plots.py`: it writes the figures next to itself, in `plots/figures/`.

```
.
├── src/
│   ├── 00_smoke_test.py                # unit tests + full pipeline on synthetic data (CPU, ~5 min)
│   ├── 01_prepare_data.py              # read the provided Seismic .bin files, map token ids to the BERT vocab, cache as CSR
│   ├── 02_download_msmarco.py          # MS MARCO queries / qrels / passages (+ TREC DL 2019/2020)
│   ├── 03_check_encoder.py             # which SPLADE checkpoint reproduces the provided vectors
│   ├── 04_baseline_splade.py           # SPLADE baseline, exact search, check vs. the provided ground truth
│   ├── 05_encode_train_queries.py      # SPLADE vectors of the MS MARCO training queries
│   ├── 06_train_concept_model.py       # training: --mode frozen | e2e, InfoNCE, hard negatives
│   ├── 07_mine_negatives.py            # hard-negative mining (SPLADE or a trained model)
│   ├── 08_eval_concept_model.py        # official evaluation (dense search), concept analysis, interventions
│   ├── 08b_eval_streaming.py           # same, streaming exact search for frozen models with k >= 2048 (does not fit in 40 GB)
│   ├── 08c_eval_e2e_streaming.py       # same for e2e models; re-encodes the corpus and saves the sparse shards used by 11
│   ├── 09_lce_splade.py                # legacy: Latent Concept Expansion baseline from the earlier SPLARE project
│   ├── 10_eval_beir.py                 # zero-shot BEIR (--splade, --ckpts, --ft-splade, --hybrid)
│   ├── 11_eval_hybrid.py               # hybrids: λ chosen on validation, then dev / DL19 / DL20 (--variant two|single|now)
│   ├── 12_significance.py              # paired per-query tests: bootstrap CI + t-test + permutation test
|   ├── 13_eval_nq.py                   # zero-shot NQ (BEIR) on the pre-encoded vectors: --splade, --ckpts, --ft-splade, --hybrid
│   |
│   ├── config.py                       # paths (env vars FM_DIR, WORK_DIR)
│   ├── sparse_io.py  retrieval.py      # CSR / Seismic I/O; exact GPU search
│   ├── splade_encoder.py               # SPLADE encoder (memory-efficient pooling)
│   ├── concept_model.py                # W, InfoNCE, false-negative mask, checkpoints
│   ├── metrics.py  msmarco.py  pipeline.py  utils.py
│   ├── lce.py                          # used by the smoke test and by 09 (LCE of the earlier project)
│
├── jobs/                           # generic Slurm job scripts (gpu.sbatch, cpu.sbatch)
├── docs/GUIDE.md                   # step-by-step guide: how every result was produced
├── docs/RESULTS.md                 # all result tables, final checkpoints and hyper-parameters
├── plots/                          # concept_model_plots.py and the figures (plots/figures/)
├── requirements.txt                # direct dependencies
└── requirements_env.txt            # exact versions of the environment used for all experiments
```

Raw logs and per-query results are **not** distributed with the repository.

The scripts themselves write under `$WORK_DIR`. Checkpoints, embeddings and caches (about 260 GB) are **not** distributed.

## Quickstart

```bash
conda create -n splade python=3.11 && conda activate splade
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8 build used on the cluster
pip install -r requirements.txt            # or: pip install -r requirements_env.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -u src/00_smoke_test.py                 # should end with: ALL TESTS PASSED (9)
```

The pipeline needs the pre-encoded MS MARCO SPLADE vectors in Seismic format (`documents.bin`, `queries.bin`, ...),
which were provided to us and are **not public**, so the full pipeline cannot be re-run from scratch without them
(you can regenerate them by encoding MS MARCO with `naver/splade-cocondenser-ensembledistil`). Paths are read from
environment variables:

```bash
export FM_DIR=/path/to/seismic/encoded/msmarco      # read-only input
export WORK_DIR=/path/to/large/scratch/splade_concept   # caches, checkpoints, results (several hundred GB for all experiments)
```

Every long step is a Slurm job. The complete sequence, with the exact commands, is in [`docs/GUIDE.md`](docs/GUIDE.md).
The figures are produced with `python plots/concept_model_plots.py` (numbers are written inside the script).

NQ uses the pre-encoded SPLADE vectors tuskanny/sparse-nq-splade-cocondenser.

## Legacy note on LCE

`src/09_lce_splade.py` and `src/lce.py` implement Latent Concept Expansion (a training-free expansion with a corpus co-occurrence matrix), the baseline of the SPLARE project.
They are not part of the results above. They are kept because `src/00_smoke_test.py` imports `src/lce.py` and runs `src/09_lce_splade.py`, and could always be useful in future work.

## Acknowledgements

This work builds on SPLADE (Formal et al., SIGIR 2021) and uses the MS MARCO and BEIR datasets.

## License

MIT, see [LICENSE](LICENSE).

## Author

Leonardo Santoro, Sapienza University of Rome.
