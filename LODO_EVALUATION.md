# Three-domain LODO evaluation — completed

All three six-epoch diagnostics completed from the original foundation. Accepted submission1003877 and the production model/calibration remain unchanged; no competition resubmission occurred.

## Consolidated source-to-target comparison

| Target | Source raw | Source calibrated | Target raw | Target calibrated | Raw gap | Calibrated gap | Target calibration gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 0.841522 | 0.850671 | 0.786128 | 0.797104 | 0.055394 | 0.053567 | 0.010976 |
| PETA | 0.849077 | 0.860265 | 0.769342 | 0.777228 | 0.079735 | 0.083037 | 0.007886 |
| Market1501 | 0.856139 | 0.865920 | 0.813343 | 0.817448 | 0.042796 | 0.048473 | 0.004104 |

| Setting | Macro held-out score | Worst-domain score | Macro mean-image-F1 score |
| --- | --- | --- | --- |
| target_raw | 0.789605 | 0.769342 | 0.787240 |
| target_calibrated | 0.797260 | 0.777228 | 0.794709 |

Macro averages weight the three excluded datasets equally; these are three independently trained models, not a deployable ensemble. Source-selection scores retain checkpoint-selection optimism. Source and target metrics use the same indicated convention.

## Target metrics

| Target | Setting | Challenge_Avg | mA | Instance F1 | Precision | Recall | Label F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | raw | 0.786128 | 0.759461 | 0.814737 | 0.800748 | 0.829223 | 0.528207 |
| PA100k | calibrated | 0.797104 | 0.786334 | 0.808174 | 0.772357 | 0.847475 | 0.523229 |
| PETA | raw | 0.769342 | 0.746214 | 0.793949 | 0.802023 | 0.786037 | 0.545401 |
| PETA | calibrated | 0.777228 | 0.769719 | 0.784886 | 0.767434 | 0.803150 | 0.537806 |
| Market1501 | raw | 0.813343 | 0.788681 | 0.839598 | 0.850232 | 0.829227 | 0.529575 |
| Market1501 | calibrated | 0.817448 | 0.800970 | 0.834618 | 0.821221 | 0.848460 | 0.527449 |

## Scorer sensitivity

| Target | Setting | Mean image F1 / mean domain H | Mean image F1 / H macro | Also safe-zero support mA |
| --- | --- | --- | --- | --- |
| PA100k | raw | 0.783551 | 0.783551 | 0.783551 |
| PA100k | calibrated | 0.794282 | 0.794282 | 0.794282 |
| PETA | raw | 0.766532 | 0.766532 | 0.766532 |
| PETA | calibrated | 0.774298 | 0.774298 | 0.774298 |
| Market1501 | raw | 0.811638 | 0.811638 | 0.787495 |
| Market1501 | calibrated | 0.815546 | 0.815546 | 0.790685 |

Exact private organizer edge cases remain unavailable; neither alternative is asserted to reproduce them. Calibration and checkpoint choice used the original provisional objective and were fixed before target inference.

## Source supervision

| Target excluded | Zero source positives | Zero source negatives | Source positives1–20 |
| --- | --- | --- | --- |
| PA100k | none | none | LowerBody-Color-Orange |
| PETA | none | none | none |
| Market1501 | none | none | none |

Support is measured from source training only. A zero-positive source attribute has no supervised positive examples, and sparse target positives make estimates noisy. This separates inadequate supervision from failures despite abundant source labels; it does not identify a causal visual mechanism.

## Attribute groups

| Group | PA100k | PETA | Market1501 | Bottom3 count |
| --- | --- | --- | --- | --- |
| age | 0.693444 | 0.670749 | 0.775520 | 3 |
| gender | 0.911596 | 0.869989 | 0.915059 | 0 |
| hair | 0.804256 | 0.833757 | 0.796775 | 0 |
| upper_length | 0.889959 | 0.875832 | 0.840400 | 0 |
| upper_colors | 0.813412 | 0.796093 | 0.868329 | 0 |
| lower_length | 0.882467 | 0.870236 | 0.877795 | 0 |
| lower_colors | 0.759247 | 0.772342 | 0.748215 | 2 |
| lower_type | 0.868165 | 0.674816 | 0.912073 | 1 |
| accessories | 0.733598 | 0.697666 | 0.701956 | 3 |

The full target raw/calibrated TP/TN/FP/FN, P/R/F1/mA and source/target support are in reports/lodo/results/PER_ATTRIBUTE_TARGET_ERRORS.csv. Group mA omits unsupported target attributes; safe-zero sensitivity above exposes that choice.

## Convergence and compute

| Target | Best source checkpoint | Sampled images | Attempted updates | Successful updates | AMP skips | Epoch compute hours | Last3 source range | Convergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 6 raw | 112008 | 1752 | 1752 | 0 | 0.378 | 0.002784 | not established |
| PETA | 5 ema | 534006 | 8346 | 8344 | 2 | 1.118 | 0.002996 | not established |
| Market1501 | 5 raw | 526014 | 8220 | 8217 | 3 | 1.127 | 0.000896 | heuristic plateau |

Complete suite elapsed wall time: 1.632hours, including setup/staging/checkpoint I/O. Epoch timers exclude checkpoint save overhead; per-fold post-training evaluation times are retained in EVALUATION.json. A last3 source range below.002 with an earlier best is only a descriptive plateau heuristic. Six epochs and different update counts cannot establish full convergence or rule out EMA lag.

## Calibration transfer

Source-only calibration gains are positive in 3/3 targets; per-fold gains are PA100k:+0.010976, PETA:+0.007886, Market1501:+0.004104. These are untouched target measurements under a fixed checkpoint/threshold lock, not target-fitted offsets.

## Interpretation and evidence ranking

1. **Measured cross-domain transfer weakness is the leading finding.** Source-to-target calibrated gaps are 5.36 points for PA100K, 8.30 for PETA and 4.85 for Market1501. The worst raw held-out score (PETA, .76934) is near the original official .77045, showing that permitted source-domain shifts can produce a gap of comparable scale. This is not evidence that PETA identifies the hidden domain or that the datasets have equivalent difficulty. The hidden gap cannot be decomposed causally from these runs.
2. **Attribute imbalance and label-distribution/semantic shift contribute plausible, distinct limitations.** All 40 attributes have both positive and negative source examples in every fold. PA-held-out lower-orange has only four positive source examples; its zero target recall is consistent with inadequate rare-label supervision. Other failures persist with substantial supervision: Market-held-out normal glasses has 13,038 source positives but calibrated target recall .129. Such failures cannot be explained by missing source labels alone. Age/accessory groups repeatedly transfer poorly; rare colors and heterogeneous 'Other' categories also fail. Annotation-policy differences versus visual feature shift are not separately identifiable here.
3. **Calibration is a secondary tradeoff, not the dominant demonstrated cause.** It improves held-out Challenge_Avg on all three targets by .41–1.10 points and macro by .77 points, while increasing FP and reducing FN. Instance-F1 and label-F1 need not improve when the mA-focused objective improves. Some attributes worsen (Market normal glasses recall falls from .206 to .129); there is no universal FP-only or FN-only failure. No target thresholds were refitted.
4. **Pretrained representation degradation and the best fine-tuning strategy remain untested mechanisms.** There is no frozen-backbone or pretrained/fine-tuned interpolation counterfactual. Source improvement alongside a target gap does not establish that visual representations were damaged. A/B may help, but claiming that these runs prove them would be unsupported. No interpolation was implemented or deployed.

The threefold target macro is .79726 after calibration, still below the roughly .865 known-domain validation score. This is not an apples-to-apples estimate of the hidden-score gap: training domains, examples, updates, six versus twelve epochs, raw/EMA selection and scorer conventions differ. PA's best is at epoch6 and its source score still improves; its smaller update budget and EMA lag are especially important. PETA is not demonstrably converged; Market shows only a descriptive plateau. AMP skipped 0/2/3 updates, under .04% per fold; no infrastructure failure or missing fold is indicated.

| Candidate family | Evidence and decision |
| --- | --- |
| A: preserve pretrained representations | No counterfactual measurement; plausible hypothesis only. |
| B: change fine-tuning | Unequal updates/EMA and incomplete convergence confound the mechanism; not selected now. |
| C: attribute imbalance | Clear rare-label limitations, but abundant-label transfer failures rule out a complete explanation. Retain current loss for the proposed comparison. |
| D: domain sampling | Measured 74–75% large-source exposure and repeated transfer gaps; selected as one bounded testable change. No simultaneous augmentation change. |
| E: calibration stability | Positive aggregate transfer in all3; per-attribute regressions merit reporting, but replacing calibration is lower priority. |

## Error balance and weakest individual attributes

| Target | Setting | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- |
| PA100k | raw | 54208 | 319705 | 14159 | 11368 |
| PA100k | calibrated | 55418 | 316458 | 17406 | 10158 |
| PETA | raw | 35544 | 223770 | 9098 | 10108 |
| PETA | calibrated | 36351 | 221168 | 11700 | 9301 |
| Market1501 | raw | 105149 | 511887 | 18982 | 22302 |
| Market1501 | calibrated | 107612 | 506162 | 24707 | 19839 |

Counts pool valid attribute decisions within a target; they are not instance-average scores.

| Target | Attribute | Source positives | Target positives | mA | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | LowerBody-Color-Purple | 56 | 5 | 0.4996 | 0.0000 | 0.0000 | 0.0000 |
| PA100k | LowerBody-Color-Orange | 4 | 65 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| PA100k | LowerBody-Color-Other | 436 | 256 | 0.5414 | 0.4231 | 0.0859 | 0.1429 |
| PA100k | UpperBody-Color-Other | 956 | 1053 | 0.6107 | 0.7081 | 0.2327 | 0.3503 |
| PA100k | Accessory-Glasses-Sun | 458 | 96 | 0.6210 | 0.0836 | 0.2708 | 0.1278 |
| PA100k | UpperBody-Color-Purple | 658 | 143 | 0.6610 | 0.1548 | 0.3497 | 0.2146 |
| PETA | UpperBody-Color-Other | 5390 | 4 | 0.4738 | 0.0000 | 0.0000 | 0.0000 |
| PETA | LowerBody-Color-Purple | 130 | 21 | 0.4999 | 0.0000 | 0.0000 | 0.0000 |
| PETA | LowerBody-Color-Yellow | 435 | 16 | 0.5613 | 0.1111 | 0.1250 | 0.1176 |
| PETA | Age-Young | 4285 | 237 | 0.5614 | 0.5455 | 0.1266 | 0.2055 |
| PETA | LowerBody-Color-Pink | 1235 | 17 | 0.6162 | 0.1667 | 0.2353 | 0.1951 |
| PETA | Accessory-Glasses-Sun | 542 | 267 | 0.6172 | 0.8514 | 0.2360 | 0.3695 |
| Market1501 | LowerBody-Color-Yellow | 291 | 16 | 0.4951 | 0.0000 | 0.0000 | 0.0000 |
| Market1501 | Accessory-Glasses-Sun | 808 | 364 | 0.5526 | 0.3478 | 0.1099 | 0.1670 |
| Market1501 | Accessory-Glasses-Normal | 13038 | 2855 | 0.5611 | 0.8084 | 0.1285 | 0.2218 |
| Market1501 | Hair-Length-Bald | 1229 | 175 | 0.6189 | 0.1648 | 0.2514 | 0.1991 |
| Market1501 | LowerBody-Color-Other | 1843 | 987 | 0.6235 | 0.3894 | 0.2746 | 0.3220 |
| Market1501 | UpperBody-Color-Other | 4440 | 1459 | 0.6387 | 0.6594 | 0.2920 | 0.4048 |

Six weakest supported target attributes per fold by calibrated mA. Sparse target positives have high estimation uncertainty; zero-positive Market color attributes are covered by the separate scorer sensitivity, not ranked as measured positive recall.

## Actual source-only trajectories

| Target excluded | Epoch | Training loss | Selection score | Selected weights | Epoch seconds |
| --- | --- | --- | --- | --- | --- |
| PA100k | 1 | 0.373816 | 0.791867 | raw | 186.8 |
| PA100k | 2 | 0.223183 | 0.820914 | raw | 236.5 |
| PA100k | 3 | 0.187926 | 0.832830 | raw | 235.1 |
| PA100k | 4 | 0.171036 | 0.838934 | raw | 233.2 |
| PA100k | 5 | 0.162454 | 0.838738 | raw | 233.4 |
| PA100k | 6 | 0.157667 | 0.841522 | raw | 235.1 |
| PETA | 1 | 0.258061 | 0.831498 | raw | 490.0 |
| PETA | 2 | 0.165020 | 0.840584 | raw | 707.9 |
| PETA | 3 | 0.139557 | 0.843531 | raw | 710.7 |
| PETA | 4 | 0.122470 | 0.846081 | raw | 706.1 |
| PETA | 5 | 0.113216 | 0.849077 | ema | 706.4 |
| PETA | 6 | 0.109612 | 0.848233 | ema | 705.3 |
| Market1501 | 1 | 0.262286 | 0.820309 | raw | 472.3 |
| Market1501 | 2 | 0.170331 | 0.840845 | raw | 715.6 |
| Market1501 | 3 | 0.146127 | 0.849732 | raw | 717.5 |
| Market1501 | 4 | 0.130526 | 0.855407 | raw | 715.5 |
| Market1501 | 5 | 0.122869 | 0.856139 | raw | 718.5 |
| Market1501 | 6 | 0.119624 | 0.855243 | raw | 718.6 |

Source/group comparison is in reports/lodo/results/GROUP_SOURCE_TARGET_COMPARISON.csv. Source groups average the two source datasets equally; unsupported attributes can differ across domains, so group gaps are descriptive.

## Staging, concurrency and preservation receipt

GPU-session archive staging verified all131076 paths in **32.96s**, under120s/30s limits; no mounted per-image copy/hash/decode loop. CPU preflight was34.42s. Source-only concurrent probes retained92.8% of PA solo throughput and95.7% of PETA solo throughput. Separate CUDA affinity0/1, workers2 and threads2 were used; Market reusedGPU0 after PA.

| Probe | Training images/s | CPU loader images/s | Peak allocated GiB | Physical GPU |
| --- | --- | --- | --- | --- |
| concurrent_PA100k | 138.38 | 340.58 | 1.240 | 0 |
| concurrent_PETA | 147.05 | 419.68 | 1.240 | 1 |
| solo_Market1501_Market1501 | 146.91 | 376.96 | 1.240 | 0 |
| solo_PA100k_PA100k | 149.12 | 424.36 | 1.240 | 0 |
| solo_PETA_PETA | 153.63 | 429.32 | 1.240 | 1 |

These are48-batch source probes, not epoch throughput or deployment timing. Full six-epoch timings above include validation; startup/checkpoint overhead is included in the1.632-hour suite wall time. Provider quota changed from14.132 to12.50 remaining, consistent with approximately1.63 billed session hours, not twice the wall time.38 Kaggle tests passed, zero failed/skipped. Actual Torch2.11/CUDA12.8 runtime matched reused parity evidence; this is not a new torch2.4 GPU parity test.

Canonical submission1003877 ZIP/best/last/calibration remain preserved. Diagnostic best/last recovery checkpoints remain in the completed private Kaggle kernel outputs; local independent verification checks their server-bound lock but does not claim to rehash undownloaded weights. Small logs/configs/CSV/NPZ are retained privately in outputs/lodo/threefold-v2-results. Public Git contains aggregate metrics, curves, configs and reproducibility receipts, not images, weights, sample predictions or annotation CSVs.

## Single recommended next change

**Recommend D: test equal probability per source domain in the existing sampler.** Change only the domain exposure, from the present square-root domain-size mass to 50/50 in a two-source fold (and one-third each in a later all-domain model). Keep architecture, augmentation, weighted BCE, optimizer, schedule, EMA, preprocessing and seed fixed. This is a recommendation for a controlled comparison, not a demonstrated improvement or an authorization to train/deploy it.

The observed sampler assigns PA100K 73.76% of draws in the PETA-held-out fold and 75.12% in the Market-held-out fold. Both have substantial source-target gaps despite over 8,200 updates. Age and accessories are bottom-three groups in all three targets, including failures with thousands of positive source examples. Domain-balanced exposure directly addresses a measured source imbalance without introducing a new architecture or changing calibration, which already transfers positively in all three folds. The PA-held-out sampler is already near 50/50 and still has a gap: sampling imbalance cannot be the whole explanation, and this experiment does not prove D will outperform A/B/C.

**Highest-value next step:** one predeclared, source-only controlled repeat of the same threefold protocol with equal-domain sampling and the same six-epoch budgets, compared against these locked baselines. Retain a further disjoint diagnostic check set if feasible; the present target results have now been inspected and cannot be described as untouched for later method development. Do not search multiple exponents or choose a fold-specific setting. Judge the fixed proposal on macro/worst-domain transfer, with the same scorer sensitivity and no hidden leaderboard feedback. A single repeat is estimated at 1.6–2.6 GPU-session hours (allow 3 hours), based on this suite and its conservative projection. Oversampling smaller sources may slow data loading or overfit repeated examples; retain the throughput/budget watchdogs.

Only if the controlled comparison supports it should an all-domain candidate be trained from the original foundation. A 12-epoch all-domain run is estimated at 2.3–3.0 session hours plus 0.5–1.0 hours for calibration, packaging and offline verification, from current measured image throughput and validation/checkpoint overhead. Combined planning allowance is roughly 5–7 hours; estimates are not reservations or guarantees. The live quota check after completion reports **12.50 hours remaining**, leaving headroom. No further GPU job, modified model or competition upload was launched during this report.

Leaving the accepted model unchanged has a verified official score and no regression risk. D is more promising as a *next experiment* because it changes a measured exposure skew at low implementation cost; no measured result yet establishes that replacing the accepted model is preferable. Keep submission1003877 as the production baseline until a new candidate has legitimate validation and API evidence.

No A–E change was implemented by this diagnostic. Domain shift, imbalance and feature specialization can coexist. Differences between this six-epoch protocol and the original twelve-epoch all-domain model prevent attributing the original hidden score gap to one training component. No hidden test identities, images, labels or leaderboard search entered the runs.

## Isolation and reproducibility

PA CSV bytes were reused exactly; all folds match precommitted split hashes and have zero cross-partition decoded-pixel overlap. Source train, selection and calibration are disjoint, and held-out training rows are excluded entirely. Native foundation hash55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832, fixed seed2027, same SigLIP2NaFlex semantic head/loss/sampler/optimizer/schedule/AMP/EMA/256patches. All six epochs and source-only model/calibration locks were checked. Independent local recomputation reproduced raw/calibrated legacy and mean-image-F1 metrics, calibration-input hashes and available source-lock files. If weights were not downloaded, their server-side checksum lock remains explicitly marked remote-only in INDEPENDENT_VERIFICATION.json. Pixel independence does not prove identity/near-duplicate independence. Checkpoints/logs remain in private Kaggle outputs and local outputs/lodo where retrieved; large weights and sample-level arrays are not published in Git.
