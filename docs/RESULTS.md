# Results

All numbers come from the per-query files in `results/` and the job logs in `logs/`. Percentages are relative to SPLADE
in the same column. Test sets: MS MARCO dev (6,980 queries), TREC DL19 (43) and DL20 (54), BEIR (NFCorpus 323,
SciFact 300, FiQA 648 queries). Differences below about 0.02 on DL19 / DL20 are noise.

## 1. Final checkpoints

Chosen on validation (1,000 held-out training queries, SPLADE = 0.3639), never on dev or BEIR. All use `--negatives splade
--n-hard 7 --neg-rank-min 3`, temperature 1, seed 42. The init-std of the trained frozen runs is not 1/√k for k = 256 and 1024 (0.02), while it is for the others.

**Frozen** (only W trained)

| k | run | init-std | lr | epochs | best val |
|---|---|---|---|---|---|
| 256 | `frozen_k256_r2` | 0.02 | 1e-3 | 10 | 0.2870 |
| 512 | `frozen_k512_r2_is0442_lr3e4_ep5` | 0.0442 | 3e-4 | 5 | 0.3173 |
| 1024 | `frozen_k1024_r2_lr3e5` | 0.02 | 3e-5 | 2 | 0.3497 |
| 2048 | `frozen_k2048_r2_is0221_lr3e5_ep5` | 0.0221 | 3e-5 | 5 | 0.3507 |
| 4096 | `frozen_k4096_r2_is0156_lr3e5_ep5` | 0.0156 | 3e-5 | 5 | 0.3566 |
| 8192 | `frozen_k8192_r2_is0110_lr3e5_ep5` | 0.0110 | 3e-5 | 5 | 0.3692 |

**Random W** (not trained: W is left at its initialisation, init-std = 1/√k)

| k | run | init-std | lr | epochs | best val |
|---|---|---|---|---|---|
| 256 | `frozen_k256_rand` | 0.0625 | 1e-9 | 1 step | 0.2384 |
| 512 | `frozen_k512_rand` | 0.0442 | 1e-9 | 1 step | 0.3043 |
| 1024 | `frozen_k1024_rand` | 0.03125 | 1e-9 | 1 step | 0.3380 |
| 2048 | `frozen_k2048_rand` | 0.0221 | 1e-9 | 1 step | 0.3421 |
| 4096 | `frozen_k4096_rand` | 0.0156 | 1e-9 | 1 step | 0.3532 |
| 8192 | `frozen_k8192_rand` | 0.0110 | 1e-9 | 1 step | 0.3614 |

**e2e** (W and BERT trained, batch size 32, 1 epoch)

| k | run | init-std | lr W | lr BERT | best val | frozen val |
|---|---|---|---|---|---|---|
| 256 | `e2e_k256_r2_is0625_lr1e4_bb1e5_b32_ep1` | 0.0625 | 1e-4 | 1e-5 | 0.3182 | 0.2870 |
| 512 | `e2e_k512_r2_is0442_lr1e4_bb5e6_b32_ep1` | 0.0442 | 1e-4 | 5e-6 | 0.3450 | 0.3173 |
| 1024 | `e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1` | 0.03125 | 3e-5 | 5e-6 | 0.3672 | 0.3497 |
| 1024, fixed W | `e2e_k1024_r2_is03125_Wfix_bb5e6_b32_ep1` | 0.03125 | 1e-9 | 5e-6 | 0.3625 | – |
| 2048 | `e2e_k2048_r2_is0221_lr3e5_bb5e6_b32_ep1` | 0.0221 | 3e-5 | 5e-6 | 0.3668 | 0.3507 |
| 4096 | `e2e_k4096_r2_is0156_lr3e5_bb1e6_b32_ep1` | 0.0156 | 3e-5 | 1e-6 | 0.3641 | 0.3566 |
| 8192 | `e2e_k8192_r2_is0110_lr3e5_bb1e7_b32_ep1` | 0.0110 | 3e-5 | 1e-7 | 0.3790 | 0.3692 |

## 2. In domain: MS MARCO

SPLADE: dev MRR@10 0.3828, nDCG@10 0.4489, R@10 0.6697, R@100 0.9093, R@1000 0.9831; DL19 nDCG@10 0.7308, R@1000 0.8724;
DL20 nDCG@10 0.7203, R@1000 0.8989.

**Frozen, full official evaluation** (% of SPLADE)

| System | k | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|
| Frozen | 256 | 0.2789 (72.9%) | 0.3378 (75.3%) | 0.9513 (96.8%) | 0.6124 (83.8%) | 0.7650 (87.7%) | 0.5721 (79.4%) | 0.7961 (88.6%) |
| Frozen | 512 | 0.3256 (85.1%) | 0.3888 (86.6%) | 0.9722 (98.9%) | 0.6743 (92.3%) | 0.8187 (93.8%) | 0.6583 (91.4%) | 0.8585 (95.5%) |
| Frozen | 1024 | 0.3451 (90.2%) | 0.4099 (91.3%) | 0.9791 (99.6%) | 0.6916 (94.6%) | 0.8247 (94.5%) | 0.6790 (94.3%) | 0.8811 (98.0%) |
| Frozen | 2048 | 0.3683 (96.2%) | 0.4335 (96.6%) | 0.9813 (99.8%) | 0.7265 (99.4%) | 0.8585 (98.4%) | 0.6950 (96.5%) | 0.8853 (98.5%) |
| Frozen | 4096 | 0.3717 (97.1%) | 0.4387 (97.7%) | 0.9819 (99.9%) | 0.7195 (98.5%) | 0.8752 (100.3%) | 0.7021 (97.5%) | 0.8970 (99.8%) |
| Frozen | 8192 | 0.3766 (98.4%) | 0.4425 (98.6%) | 0.9818 (99.9%) | 0.7208 (98.6%) | 0.8775 (100.6%) | 0.7059 (98.0%) | 0.8948 (99.5%) |

**Random W, full official evaluation** (% of SPLADE)

| System | k | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|
| Random W | 256 | 0.2544 (66.5%) | 0.3129 (69.7%) | 0.9462 (96.2%) | 0.5946 (81.4%) | 0.7756 (88.9%) | 0.5518 (76.6%) | 0.8302 (92.4%) |
| Random W | 512 | 0.3129 (81.7%) | 0.3762 (83.8%) | 0.9697 (98.6%) | 0.6607 (90.4%) | 0.8084 (92.7%) | 0.6352 (88.2%) | 0.8625 (96.0%) |
| Random W | 1024 | 0.3454 (90.2%) | 0.4109 (91.5%) | 0.9773 (99.4%) | 0.6854 (93.8%) | 0.8468 (97.1%) | 0.6754 (93.8%) | 0.8890 (98.9%) |
| Random W | 2048 | 0.3634 (94.9%) | 0.4296 (95.7%) | 0.9809 (99.8%) | 0.7227 (98.9%) | 0.8564 (98.2%) | 0.6908 (95.9%) | 0.8862 (98.6%) |
| Random W | 4096 | 0.3691 (96.4%) | 0.4366 (97.3%) | 0.9818 (99.9%) | 0.7153 (97.9%) | 0.8684 (99.5%) | 0.7021 (97.5%) | 0.8939 (99.4%) |
| Random W | 8192 | 0.3761 (98.2%) | 0.4434 (98.8%) | 0.9822 (99.9%) | 0.7189 (98.4%) | 0.8784 (100.7%) | 0.7166 (99.5%) | 0.8981 (99.9%) |

**e2e, full official evaluation** (% of SPLADE)

| System | k | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|
| e2e | 256 | 0.3424 (89.4%) | 0.4019 (89.5%) | 0.9565 (97.3%) | 0.6414 (87.8%) | 0.7494 (85.9%) | 0.6531 (90.7%) | 0.8166 (90.8%) |
| e2e | 512 | 0.3598 (94.0%) | 0.4230 (94.2%) | 0.9729 (99.0%) | 0.6941 (95.0%) | 0.8014 (91.9%) | 0.6690 (92.9%) | 0.8349 (92.9%) |
| e2e | 1024 | 0.3731 (97.5%) | 0.4383 (97.6%) | 0.9784 (99.5%) | 0.6880 (94.1%) | 0.8216 (94.2%) | 0.6895 (95.7%) | 0.8661 (96.4%) |
| e2e, fixed W | 1024 | 0.3714 (97.0%) | 0.4355 (97.0%) | 0.9771 (99.4%) | 0.6921 (94.7%) | 0.8158 (93.5%) | 0.6930 (96.2%) | 0.8684 (96.6%) |
| e2e | 2048 | 0.3799 (99.2%) | 0.4455 (99.2%) | 0.9805 (99.7%) | 0.7166 (98.1%) | 0.8310 (95.3%) | 0.6956 (96.6%) | 0.8512 (94.7%) |
| e2e | 4096 | 0.3829 (100.0%) | 0.4477 (99.7%) | 0.9822 (99.9%) | 0.7354 (100.6%) | 0.8499 (97.4%) | 0.7025 (97.5%) | 0.8767 (97.5%) |
| e2e | 8192 | 0.3722 (97.2%) | 0.4376 (97.5%) | 0.9801 (99.7%) | 0.7323 (100.2%) | 0.8623 (98.8%) | 0.6984 (97.0%) | 0.8896 (99.0%) |

**e2e − frozen**

| k | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| validation MRR@10 | +0.031 | +0.028 | +0.0175 | +0.016 | +0.008 | +0.010 |
| dev MRR@10 | +0.064 | +0.034 | +0.028 | +0.012 | +0.011 | −0.004 |
| BEIR mean nDCG@10 | +0.002 | +0.001 | 0.000 | −0.008 | −0.002 | −0.006 |
| BEIR mean R@100 | +0.014 | +0.019 | +0.012 | −0.002 | −0.002 | −0.005 |

The gain falls with k, stays in domain, and turns negative at 8192, where validation had suggested the opposite.

## 3. Out of domain: BEIR (nDCG@10, zero-shot)

SPLADE: NFCorpus 0.3486, SciFact 0.7024, FiQA 0.3476, **mean 0.4662** (R@100 mean 0.6160).

| System | k | NFCorpus | SciFact | FiQA | Mean (% of SPLADE) |
|---|---|---|---|---|---|
| Random W | 256 | 0.2516 | 0.4941 | 0.1895 | 0.3117 (66.9%) |
| Frozen | 256 | 0.2605 | 0.5252 | 0.2211 | 0.3356 (72.0%) |
| e2e | 256 | 0.2777 | 0.5057 | 0.2283 | 0.3372 (72.3%) |
| Random W | 512 | 0.3039 | 0.5997 | 0.2752 | 0.3929 (84.3%) |
| Frozen | 512 | 0.3076 | 0.6235 | 0.2736 | 0.4016 (86.1%) |
| e2e | 512 | 0.3068 | 0.6172 | 0.2840 | 0.4027 (86.4%) |
| Random W | 1024 | 0.3264 | 0.6617 | 0.3019 | 0.4300 (92.2%) |
| Frozen | 1024 | 0.3254 | 0.6660 | 0.2968 | 0.4294 (92.1%) |
| e2e | 1024 | 0.3251 | 0.6525 | 0.3098 | 0.4291 (92.0%) |
| e2e, fixed W | 1024 | 0.3206 | 0.6553 | 0.3086 | 0.4282 (91.8%) |
| Random W | 2048 | 0.3310 | 0.6657 | 0.3143 | 0.4370 (93.7%) |
| Frozen | 2048 | 0.3330 | 0.6744 | 0.3219 | 0.4431 (95.0%) |
| e2e | 2048 | 0.3281 | 0.6533 | 0.3228 | 0.4347 (93.3%) |
| Random W | 4096 | 0.3469 | 0.6910 | 0.3425 | 0.4601 (98.7%) |
| Frozen | 4096 | 0.3464 | 0.6920 | 0.3411 | 0.4598 (98.6%) |
| e2e | 4096 | 0.3449 | 0.6855 | 0.3436 | 0.4580 (98.2%) |
| Random W | 8192 | 0.3458 | 0.6936 | 0.3405 | 0.4600 (98.7%) |
| Frozen | 8192 | 0.3480 | 0.6922 | 0.3412 | 0.4605 (98.8%) |
| e2e | 8192 | 0.3453 | 0.6813 | 0.3358 | 0.4541 (97.4%) |

For random W and frozen, the share of SPLADE kept is the same in and out of domain (within 1–2 points at every k). For
e2e it is much lower out of domain at small k (89.4% in domain vs 72.3% on BEIR at k = 256): the gain is a specialisation
to MS MARCO. SciFact (scientific claims) is the most penalised and FiQA (natural-language questions, close to MS MARCO) the most favoured.
The difference between Frozen and Random W follows the same pattern in and out of domain: training W helps clearly at k = 256, a little at k = 512, and from k = 1024 upwards the learned W is almost indistinguishable from a random one.

## 4. Hybrids with SPLADE

A hybrid mixes SPLADE and the Concept Model in a single score: (1−λ)·SPLADE(q)·SPLADE(d) + λ·g(q)·g(d), where λ = 0 is SPLADE alone and λ = 1 is the Concept Model alone.

λ\* is chosen on 1,000 validation queries with a full-corpus search over 21 values (0, 0.05, …, 1).

Variants of the hybrid, from the original one to the re-trained encoder alone. `SPLADE_θ` is the encoder re-trained end-to-end (e2e checkpoints), and `g(x) = W·SPLADE_θ(x)`.

| Variant | Score | Space (what must be stored) | What changes |
|---|---|---|---|
| `hyb` | `(1−λ)·SPLADE(q)·SPLADE(d) + λ·g(q)·g(d)` | • Index of the original SPLADE<br>• Sparse vectors of the re-trained encoder θ (needed for the term with W) | The original hybrid. |
| `hyb1` | `(1−λ)·SPLADE_θ(q)·SPLADE_θ(d) + λ·g(q)·g(d)` | • ~~Index of the original SPLADE~~<br>• Sparse vectors of the re-trained encoder θ | The original SPLADE is no longer used: it is replaced by `SPLADE_θ`, the re-trained encoder. |
| `hybnoW` | `(1−λ)·SPLADE(q)·SPLADE(d) + λ·SPLADE_θ(q)·SPLADE_θ(d)` | • Index of the original SPLADE<br>• Sparse vectors of the re-trained encoder θ | The second term does not use `g` (that is, `W·SPLADE_θ`) but `SPLADE_θ` directly, without W. |
| `ftsplade` (ft = fine-tuned SPLADE) | `SPLADE_θ(q)·SPLADE_θ(d)` | • ~~Index of the original SPLADE~~<br>• Sparse vectors of the re-trained encoder θ | Same as `hyb1` with λ = 0: the re-trained e2e encoder used directly as a plain SPLADE. |

**Hybrids, in domain: dev MRR@10 (% of SPLADE)**:

| k | Frozen hybrid | λ\* | e2e hybrid (`hyb`) | λ\* | `hybnoW` | λ\* | `ftsplade` |
|---|---|---|---|---|---|---|---|
| 256 | 0.3835 (100.2%) | 0.05 | 0.3901 (101.9%) | 0.20 | **0.3920 (102.4%)** | 0.40 | 0.3798 (99.2%) |
| 512 | 0.3815 (99.7%) | 0.15 | 0.3873 (101.2%) | 0.40 | 0.3910 (102.1%) | 0.45 | 0.3856 (100.7%) |
| 1024 | 0.3779 (98.7%) | 0.45 | 0.3855 (100.7%) | 0.75 | 0.3915 (102.3%) | 0.50 | 0.3884 (101.5%) |
| 2048 | 0.3828 (100.0%) | 0.00 | 0.3881 (101.4%) | 0.40 | 0.3908 (102.1%) | 0.90 | 0.3900 (101.9%) |
| 4096 | 0.3831 (100.1%) | 0.10 | 0.3880 (101.4%) | 0.30 | 0.3887 (101.5%) | 0.90 | 0.3879 (101.3%) |
| 8192 | 0.3806 (99.4%) | 0.85 | 0.3722 (97.2%) | 1.00 | 0.3847 (100.5%) | 0.40 | 0.3832 (100.1%) |

SPLADE alone: 0.3828.

**Hybrids, out of domain: BEIR mean nDCG@10 (% of SPLADE)**:

| k | Frozen hybrid | λ\* | e2e hybrid (`hyb`) | λ\* | `hybnoW` | λ\* | `ftsplade` |
|---|---|---|---|---|---|---|---|
| 256 | 0.4659 (99.9%) | 0.05 | 0.4646 (99.7%) | 0.20 | **0.4676 (100.3%)** | 0.40 | 0.4248 (91.1%) |
| 512 | 0.4642 (99.6%) | 0.15 | 0.4608 (98.8%) | 0.40 | 0.4646 (99.7%) | 0.45 | 0.4419 (94.8%) |
| 1024 | 0.4604 (98.8%) | 0.45 | 0.4510 (96.7%) | 0.75 | 0.4654 (99.8%) | 0.50 | 0.4513 (96.8%) |
| 2048 | 0.4662 (100.0%) | 0.00 | 0.4630 (99.3%) | 0.40 | 0.4555 (97.7%) | 0.90 | 0.4517 (96.9%) |
| 4096 | 0.4660 (100.0%) | 0.10 | 0.4661 (100.0%) | 0.30 | 0.4617 (99.0%) | 0.90 | 0.4598 (98.6%) |
| 8192 | 0.4617 (99.0%) | 0.85 | 0.4541 (97.4%) | 1.00 | 0.4646 (99.7%) | 0.40 | 0.4606 (98.8%) |

SPLADE alone: 0.4662.

## 5. Significance tests

All tests are paired and two-sided: Δ = B − A, 95% bootstrap confidence interval, permutation p-value (10,000 sign flips).

**Which variant is better?** Differences in dev MRR@10 (6,980 queries) between the variants of the hybrid, k by k. A negative `hyb` − `hybnoW` means that the hybrid without W is better; "yes" means p < 0.05.

| k | `hyb` − `hybnoW` | `hyb1` − SPLADE | `ftsplade` − SPLADE |
|---|---|---|---|
| 256 | −0.0019, p = 0.13 (no) | −0.0025, p = 0.38 (no) | −0.0030, p = 0.28 (no) |
| 512 | −0.0038, p = 0.009 (yes: without W is better) | +0.0028, p = 0.26 (no) | +0.0028, p = 0.26 (no) |
| 1024 | −0.0060, p = 0.003 (yes: without W is better) | +0.0068, p = 0.006 (yes) | +0.0056, p = 0.019 (yes) |
| 2048 | −0.0028, p = 0.11 (no) | +0.0072, p = 0.002 (yes) | +0.0072, p = 0.002 (yes) |
| 4096 | −0.0007, p = 0.68 (no) | +0.0066, p = 0.002 (yes) | +0.0051, p = 0.013 (yes) |
| 8192 | −0.0125, p = 0.0001 (yes: without W is better) | −0.0106, p = 0.0001 (yes, but worse) | +0.0004, p = 0.81 (no) |

The hybrid without W is never worse than the one with W: the Concept Model helps by training the encoder, not at search time. The re-trained encoder alone (`ftsplade`) and `hyb1` beat SPLADE significantly in domain at k = 1024–4096 and are not better at the extremes; at k = 8192 `hyb1` is worse because λ\* = 1 leaves only the e2e model.

**The best hybrid against SPLADE, in domain** (`hybnoW`, k = 256, λ\* = 0.40; dev, 6,980 queries). The gain is in the ordering of the top results: MRR@10 and nDCG@10 improve significantly, while recall at 1000 does not change.

| Metric | SPLADE | `hybnoW` | Δ | 95% CI | p (perm.) |
|---|---|---|---|---|---|
| MRR@10 | 0.3828 | 0.3920 | **+0.0092** | [+0.0063, +0.0123] | 0.0001 |
| nDCG@10 | 0.4489 | 0.4577 | **+0.0089** | [+0.0063, +0.0114] | 0.0001 |
| R@1000 | 0.9831 | 0.9841 | +0.0010 | [−0.0001, +0.0022] | 0.10 |

**The same system out of domain**, per BEIR dataset. No consistent gain: only FiQA improves (nDCG@10 and MRR@10), NFCorpus is slightly worse in precision but better in recall, and SciFact does not change.

*NFCorpus (323 queries)*

| Metric | SPLADE | `hybnoW` | Δ | 95% CI | p (perm.) |
|---|---|---|---|---|---|
| MRR@10 | 0.5782 | 0.5679 | −0.0104 | [−0.0213, −0.0001] | 0.053 |
| nDCG@10 | 0.3486 | 0.3447 | −0.0039 | [−0.0080, +0.0000] | 0.056 |
| R@1000 | 0.5907 | 0.6087 | **+0.0179** | [+0.0052, +0.0319] | **0.006** |

*SciFact (300 queries)*

| Metric | SPLADE | `hybnoW` | Δ | 95% CI | p (perm.) |
|---|---|---|---|---|---|
| MRR@10 | 0.6699 | 0.6742 | +0.0042 | [−0.0088, +0.0173] | 0.54 |
| nDCG@10 | 0.7024 | 0.7047 | +0.0023 | [−0.0084, +0.0130] | 0.69 |
| R@1000 | 0.9867 | 0.9933 | +0.0067 | [+0.0000, +0.0167] | 0.50 |

*FiQA (648 queries)*

| Metric | SPLADE | `hybnoW` | Δ | 95% CI | p (perm.) |
|---|---|---|---|---|---|
| MRR@10 | 0.4180 | 0.4277 | **+0.0097** | [+0.0011, +0.0186] | **0.030** |
| nDCG@10 | 0.3476 | 0.3534 | **+0.0058** | [+0.0009, +0.0107] | **0.022** |
| R@1000 | 0.8402 | 0.8460 | +0.0057 | [−0.0014, +0.0128] | 0.12 |


## 6. Space (8.84M passages, raw sizes)

| System | Stored per document | Total |
|---|---|---|
| SPLADE | sparse index, ~120 non-zeros | ~8.5 GB |
| Concept Model alone | k fp16 numbers | 4.5 / 9.1 / 18.1 / 36.2 / 72.4 / 144.9 GB for k = 256 … 8192 |
| Frozen hybrid | SPLADE index + W once (30,522 × k fp32, 31 MB to 1.0 GB) | 8.5–9.5 GB |
| e2e hybrid, `hybnoW` | SPLADE index + sparse index of the re-trained encoder (102 / 115 / 145 / 166 / 149 / 105 non-zeros) | 15.8 / 16.7 / 18.8 / 20.3 / 19.1 / 16.0 GB |

| System | Stored per document | Bytes/doc (raw) | Total (8.84M docs) | vs. SPLADE |
|---|---|---|---|---|
| SPLADE | original sparse index (~120 non-zeros) | ~1.0 KB | ~8.5 GB | ×1.00 |
| frozen / e2e 256 | dense vector g(d), 256 fp16 numbers | 0.5 KB | ~4.5 GB | ×0.53 |
| frozen / e2e 512 | dense vector, 512 fp16 numbers | 1.0 KB | ~9.1 GB | ×1.07 |
| frozen / e2e 1024 | dense vector, 1024 fp16 numbers | 2.0 KB | ~18.1 GB | ×2.1 |
| frozen / e2e 2048 | dense vector, 2048 fp16 numbers | 4.1 KB | ~36.2 GB | ×4.3 |
| frozen / e2e 4096 | dense vector, 4096 fp16 numbers | 8.2 KB | ~72.4 GB | ×8.5 |
| frozen / e2e 8192 | dense vector, 8192 fp16 numbers | 16.4 KB | ~144.9 GB | ×17.0 |
| Frozen hybrid (every k) | SPLADE index only (+ W once: 30,522 × k fp32, 31 MB to 1.0 GB) | ~1.0 KB | ~8.5 GB | ×1.00 |
| e2e hybrid 256 | SPLADE index + sparse index of the e2e encoder (~102 non-zeros; 7.3 GB) | ~1.8 KB | ~15.8 GB | ×1.86 |
| e2e hybrid 512 | SPLADE index + sparse index of the e2e encoder (~115 non-zeros; 8.2 GB) | ~1.9 KB | ~16.7 GB | ×1.97 |
| e2e hybrid 1024 | SPLADE index + sparse index of the e2e encoder (144.7 non-zeros; 10.3 GB) | ~2.1 KB | ~18.8 GB | ×2.2 |
| e2e hybrid 2048 | SPLADE index + sparse index of the e2e encoder (166.4 non-zeros; 11.8 GB) | ~2.3 KB | ~20.3 GB | ×2.4 |
| e2e hybrid 4096 | SPLADE index + sparse index of the e2e encoder (148.6 non-zeros; 10.6 GB) | ~2.2 KB | ~19.1 GB | ×2.2 |
| e2e hybrid 8192 | SPLADE index + sparse index of the e2e encoder (104.6 non-zeros; 7.5 GB) | ~1.8 KB | ~16.0 GB | ×1.88 |

A sparse entry takes about 8 bytes (int32 id + fp32 weight) plus a per-document pointer. No Concept-Model variant is smaller
than SPLADE at equal quality, because SPLADE stores only ~120 values per document while the dense model stores all k.
bf16 has the same size as fp16; int8 would halve it and was not tried.
