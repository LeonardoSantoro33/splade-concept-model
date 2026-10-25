# Results

All numbers come from the per-query files in `results/` and the job logs in `logs/`. Percentages are relative to SPLADE
in the same column. Test sets: MS MARCO dev (6,980 queries), TREC DL19 (43) and DL20 (54), BEIR (NFCorpus 323,
SciFact 300, FiQA 648 queries). Differences below about 0.02 on DL19 / DL20 are noise.

## 1. Final checkpoints

Chosen on validation (1,000 held-out training queries, SPLADE = 0.3639), never on dev or BEIR. All use `--negatives splade
--n-hard 7 --neg-rank-min 3`, temperature 1, seed 42.

**Frozen** (only W trained)

| k | run (`models/<run>/best`) | init-std | lr | epochs | best val |
|---|---|---|---|---|---|
| 256 | `frozen_k256_r2` | 0.02 | 1e-3 | 10 | 0.2870 |
| 512 | `frozen_k512_r2_is0442_lr3e4_ep5` | 0.0442 | 3e-4 | 5 | 0.3173 |
| 1024 | `frozen_k1024_r2_lr3e5` | 0.02 | 3e-5 | 2 | 0.3497 |
| 2048 | `frozen_k2048_r2_is0221_lr3e5_ep5` | 0.0221 | 3e-5 | 5 | 0.3507 |
| 4096 | `frozen_k4096_r2_is0156_lr3e5_ep5` | 0.0156 | 3e-5 | 5 | 0.3566 |
| 8192 | `frozen_k8192_r2_is0110_lr3e5_ep5` | 0.0110 | 3e-5 | 5 | 0.3692 |

**Random W** (not trained: W is left at its initialisation, init-std = 1/√k)

| k | run (`models/<run>/best`) | init-std | lr | epochs | best val |
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

**dev MRR@10 (% of SPLADE)**

| k | bytes/doc | Random W | Frozen | e2e |
|---|---|---|---|---|
| 256 | 512 | 0.2544 (66.5%) | 0.2789 (72.9%) | 0.3424 (89.4%) |
| 512 | 1,024 | 0.3129 (81.7%) | 0.3256 (85.1%) | 0.3598 (94.0%) |
| 1024 | 2,048 | 0.3454 (90.2%) | 0.3451 (90.2%) | 0.3731 (97.5%) |
| 2048 | 4,096 | 0.3634 (94.9%) | 0.3683 (96.2%) | 0.3799 (99.2%) |
| 4096 | 8,192 | 0.3691 (96.4%) | 0.3717 (97.1%) | **0.3829 (100.0%)** |
| 8192 | 16,384 | 0.3761 (98.2%) | 0.3766 (98.4%) | 0.3722 (97.2%) |

e2e with fixed W, k = 1024: 0.3714 (97.0%). About 94% of the e2e gain over the random W at k = 1024 is obtained with BERT alone.

**e2e, full official evaluation**

| System | k | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|
| e2e | 256 | 0.3424 | 0.4019 | 0.9565 | 0.6414 | 0.7494 | 0.6531 | 0.8166 |
| e2e | 512 | 0.3598 | 0.4230 | 0.9729 | 0.6941 | 0.8014 | 0.6690 | 0.8349 |
| e2e | 1024 | 0.3731 | 0.4383 | 0.9784 | 0.6880 | 0.8216 | 0.6895 | 0.8661 |
| e2e, fixed W | 1024 | 0.3714 | 0.4355 | 0.9771 | 0.6921 | 0.8158 | 0.6930 | 0.8684 |
| e2e | 2048 | 0.3799 | 0.4455 | 0.9805 | 0.7166 | 0.8310 | 0.6956 | 0.8512 |
| e2e | 4096 | 0.3829 | 0.4477 | 0.9822 | 0.7354 | 0.8499 | 0.7025 | 0.8767 |
| e2e | 8192 | 0.3722 | 0.4376 | 0.9801 | 0.7323 | 0.8623 | 0.6984 | 0.8896 |

(Streaming evaluation `08c` vs. dense `08` at k = 1024: differences ≤ 0.0006, fp16 rounding.)

**e2e − frozen**

| k | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| validation MRR@10 | +0.031 | +0.028 | +0.0175 | +0.016 | +0.008 | +0.010 |
| dev MRR@10 | +0.064 | +0.034 | +0.028 | +0.012 | +0.011 | −0.004 |
| BEIR mean nDCG@10 | +0.002 | +0.001 | 0.000 | −0.008 | −0.002 | −0.006 |
| BEIR mean R@100 | +0.014 | +0.019 | +0.012 | −0.002 | −0.002 | −0.005 |

The gain falls with k, stays in domain, and turns negative at 8192, where validation had suggested the opposite
(the BERT learning-rate grid had to be extended three times to 1e-7, so the winner is picked among many noisy values).

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
Learned W minus random W (BEIR mean): +0.024, +0.009, −0.001, +0.006, 0.000, +0.0005 for k = 256 … 8192, the same pattern as in domain.

**Frozen hybrid vs. e2e hybrid on BEIR, per dataset (18 tests; Δ = frozen − e2e, permutation p).** Frozen is never significantly worse.
It is better in 4 of 18 cases at the 5% level: SciFact at 512 (+0.0137, p = 0.020), 1024 (+0.0193, p = 0.004) and 8192 (+0.0113, p = 0.029), FiQA at
8192 (+0.0094, p = 0.010). None survives Bonferroni (0.05 / 18 ≈ 0.0028): a consistent tendency, not strong evidence.

## 4. Hybrids with SPLADE

λ\* is chosen on 1,000 validation queries with a full-corpus search over 21 values; 8192 chose λ\* = 1 for the e2e hybrid
(SPLADE excluded: it coincides with the e2e model alone).

**dev MRR@10 and BEIR mean nDCG@10 (λ\* in brackets)**

| k | Frozen hybrid dev | e2e hybrid (`hyb`) dev | `hybnoW` dev | Frozen hybrid BEIR | `hyb` BEIR | `hybnoW` BEIR |
|---|---|---|---|---|---|---|
| 256 | 0.3835 (0.05) | 0.3901 (0.20) | **0.3920 (0.40)** | 0.4659 | 0.4646 | **0.4676** |
| 512 | 0.3815 (0.15) | 0.3873 (0.40) | 0.3910 (0.45) | 0.4642 | 0.4608 | 0.4646 |
| 1024 | 0.3779 (0.45) | 0.3855 (0.75) | 0.3915 (0.50) | 0.4604 | 0.4510 | 0.4654 |
| 2048 | 0.3828 (0.00) | 0.3881 (0.40) | 0.3908 (0.90) | 0.4662 | 0.4630 | 0.4555 |
| 4096 | 0.3831 (0.10) | 0.3880 (0.30) | 0.3887 (0.90) | 0.4660 | 0.4661 | 0.4617 |
| 8192 | 0.3806 (0.85) | 0.3722 (1.00) | 0.3847 (0.40) | 0.4617 | 0.4541 | 0.4646 |

SPLADE: dev 0.3828, BEIR 0.4662. Index sizes: frozen hybrid 8.5–9.5 GB (SPLADE index + W); e2e hybrid and `hybnoW`
15.8 / 16.7 / 18.8 / 20.3 / 19.1 / 16.0 GB.

**Significance on dev (B − A, permutation p; 6,980 queries)**

| Comparison | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| `hyb` − SPLADE | +0.0073, p = 0.0001 | +0.0045, p = 0.009 | +0.0027, p = 0.25 | +0.0053, p = 0.001 | +0.0052, p = 0.0001 | −0.0106, p = 0.0001 |
| `hyb` − `hybnoW` | −0.0019, p = 0.13 | −0.0038, p = 0.009 | −0.0060, p = 0.003 | −0.0028, p = 0.11 | −0.0007, p = 0.68 | −0.0125, p = 0.0001 |
| `hyb1` − SPLADE | −0.0025, p = 0.38 | +0.0028, p = 0.26 | +0.0068, p = 0.006 | +0.0072, p = 0.002 | +0.0066, p = 0.002 | −0.0106, p = 0.0001 |
| `ftsplade` − SPLADE | −0.0030, p = 0.28 | +0.0028, p = 0.26 | +0.0056, p = 0.019 | +0.0072, p = 0.002 | +0.0051, p = 0.013 | +0.0004, p = 0.81 |

`hybnoW` k = 256 − SPLADE: dev MRR@10 +0.0092 [+0.0063, +0.0123], p = 0.0001; nDCG@10 +0.0089, p = 0.0001; R@1000 +0.0010,
p = 0.10 (not significant). `hybnoW` k = 8192 − SPLADE: +0.0019 [+0.0002, +0.0037], p = 0.034.
The hybrid without W is never worse than the one with W; the Concept Model helps through training the encoder, not at search time.
Recall at 1000 never changes significantly: the gains are in the ordering of the top results.

**`hybnoW` k = 256 against SPLADE on BEIR, per dataset (nDCG@10):** NFCorpus −0.0039 (p = 0.056), SciFact +0.0023 (p = 0.69),
FiQA +0.0058 (p = 0.022); NFCorpus R@1000 +0.0179 (p = 0.006). Not significant after correcting for 3 (or 9) tests: the
hybrid is *at the level of* SPLADE out of domain, not better.

**The re-trained encoder alone (`ftsplade`)**, one index, plain SPLADE scoring:

| k | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| dev MRR@10 | 0.3798 | 0.3856 | 0.3884 | 0.3900 | 0.3879 | 0.3832 |
| BEIR mean nDCG@10 (% SPLADE) | 0.4248 (91.1%) | 0.4419 (94.8%) | 0.4513 (96.8%) | 0.4517 (96.9%) | 0.4598 (98.6%) | 0.4606 (98.8%) |

It beats SPLADE in domain at k = 1024–4096 (best +0.0072, +1.9%, p = 0.002, at k = 2048; this one also survives a correction for
the number of k values tested) but is below it on BEIR at every k. Index: 7.3–11.8 GB (+21% to +39% over SPLADE).

## 5. Space (8.84M passages, raw sizes)

| System | Stored per document | Total |
|---|---|---|
| SPLADE | sparse index, ~120 non-zeros | ~8.5 GB |
| Concept Model alone | k fp16 numbers | 4.5 / 9.1 / 18.1 / 36.2 / 72.4 / 144.9 GB for k = 256 … 8192 |
| Frozen hybrid | SPLADE index + W once (30,522 × k fp32, 31 MB to 1.0 GB) | 8.5–9.5 GB |
| e2e hybrid, `hybnoW` | SPLADE index + sparse index of the re-trained encoder (102 / 115 / 145 / 166 / 149 / 105 non-zeros) | 15.8 / 16.7 / 18.8 / 20.3 / 19.1 / 16.0 GB |

A sparse entry takes about 8 bytes (int32 id + fp32 weight) plus a per-document pointer. No Concept-Model variant is smaller
than SPLADE at equal quality, because SPLADE stores only ~120 values per document while the dense model stores all k.
bf16 has the same size as fp16; int8 would halve it and was not tried.
