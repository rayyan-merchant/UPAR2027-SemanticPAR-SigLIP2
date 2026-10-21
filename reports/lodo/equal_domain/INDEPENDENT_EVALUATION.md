# Three-domain LODO evaluation — completed

All three six-epoch diagnostics completed from the original foundation. Accepted submission1003877 and the production model/calibration remain unchanged; no competition resubmission occurred.

## Consolidated source-to-target comparison

| Target | Source raw | Source calibrated | Target raw | Target calibrated | Raw gap | Calibrated gap | Target calibration gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 0.841219 | 0.851418 | 0.785534 | 0.797565 | 0.055685 | 0.053853 | 0.012031 |
| PETA | 0.844951 | 0.854805 | 0.765880 | 0.776778 | 0.079071 | 0.078027 | 0.010897 |
| Market1501 | 0.859042 | 0.867008 | 0.808667 | 0.813926 | 0.050375 | 0.053082 | 0.005259 |

| Setting | Macro held-out score | Worst-domain score | Macro mean-image-F1 score |
| --- | --- | --- | --- |
| target_raw | 0.786694 | 0.765880 | 0.784301 |
| target_calibrated | 0.796089 | 0.776778 | 0.793542 |

Macro averages weight the three excluded datasets equally; these are three independently trained models, not a deployable ensemble. Source-selection scores retain checkpoint-selection optimism. Source and target metrics use the same indicated convention.

## Target metrics

| Target | Setting | Challenge_Avg | mA | Instance F1 | Precision | Recall | Label F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | raw | 0.785534 | 0.756661 | 0.816698 | 0.802739 | 0.831151 | 0.522421 |
| PA100k | calibrated | 0.797565 | 0.785098 | 0.810434 | 0.782528 | 0.840403 | 0.527603 |
| PETA | raw | 0.765880 | 0.742742 | 0.790507 | 0.800338 | 0.780914 | 0.536900 |
| PETA | calibrated | 0.776778 | 0.771586 | 0.782039 | 0.760357 | 0.804995 | 0.538717 |
| Market1501 | raw | 0.808667 | 0.781848 | 0.837392 | 0.847403 | 0.827614 | 0.524411 |
| Market1501 | calibrated | 0.813926 | 0.795027 | 0.833745 | 0.819174 | 0.848845 | 0.525343 |

## Scorer sensitivity

| Target | Setting | Mean image F1 / mean domain H | Mean image F1 / H macro | Also safe-zero support mA |
| --- | --- | --- | --- | --- |
| PA100k | raw | 0.783068 | 0.783068 | 0.783068 |
| PA100k | calibrated | 0.794900 | 0.794900 | 0.794900 |
| PETA | raw | 0.762953 | 0.762953 | 0.762953 |
| PETA | calibrated | 0.773740 | 0.773740 | 0.773740 |
| Market1501 | raw | 0.806882 | 0.806882 | 0.783165 |
| Market1501 | calibrated | 0.811986 | 0.811986 | 0.787446 |

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
| age | 0.689670 | 0.660464 | 0.723991 | 3 |
| gender | 0.912112 | 0.869544 | 0.912310 | 0 |
| hair | 0.809965 | 0.828226 | 0.782323 | 0 |
| upper_length | 0.887933 | 0.889040 | 0.853669 | 0 |
| upper_colors | 0.819661 | 0.791436 | 0.867343 | 0 |
| lower_length | 0.870826 | 0.863529 | 0.872580 | 0 |
| lower_colors | 0.750142 | 0.783417 | 0.743801 | 2 |
| lower_type | 0.875053 | 0.683483 | 0.914159 | 1 |
| accessories | 0.729279 | 0.702010 | 0.708959 | 3 |

The full target raw/calibrated TP/TN/FP/FN, P/R/F1/mA and source/target support are in reports/lodo/results/PER_ATTRIBUTE_TARGET_ERRORS.csv. Group mA omits unsupported target attributes; safe-zero sensitivity above exposes that choice.

## Convergence and compute

| Target | Best source checkpoint | Sampled images | Attempted updates | Successful updates | AMP skips | Epoch compute hours | Last3 source range | Convergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 6 raw | 112008 | 1752 | 1752 | 0 | 0.348 | 0.001940 | not established |
| PETA | 6 ema | 534006 | 8346 | 8344 | 2 | 1.105 | 0.002883 | not established |
| Market1501 | 4 raw | 526014 | 8220 | 8217 | 3 | 1.045 | 0.001602 | heuristic plateau |

Complete suite elapsed wall time: 1.519hours, including setup/staging/checkpoint I/O. Epoch timers exclude checkpoint save overhead; per-fold post-training evaluation times are retained in EVALUATION.json. A last3 source range below.002 with an earlier best is only a descriptive plateau heuristic. Six epochs and different update counts cannot establish full convergence or rule out EMA lag.

## Calibration transfer

Source-only calibration gains are positive in 3/3 targets; per-fold gains are PA100k:+0.012031, PETA:+0.010897, Market1501:+0.005259. These are untouched target measurements under a fixed checkpoint/threshold lock, not target-fitted offsets.

## Single recommended next change

**Reviewed outcome: retain the original square-root-domain sampler.** The equal-domain intervention failed the predeclared macro/worst-domain criteria. See LODO_SAMPLING_COMPARISON.md for the controlled comparison, group/trajectory review and deadline plan. No further training or submission was launched.

No A–E change was implemented by this diagnostic. Domain shift, imbalance and feature specialization can coexist. Differences between this six-epoch protocol and the original twelve-epoch all-domain model prevent attributing the original hidden score gap to one training component. No hidden test identities, images, labels or leaderboard search entered the runs.

## Isolation and reproducibility

PA CSV bytes were reused exactly; all folds match precommitted split hashes and have zero cross-partition decoded-pixel overlap. Source train, selection and calibration are disjoint, and held-out training rows are excluded entirely. Native foundation hash55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832, fixed seed2027, same SigLIP2NaFlex semantic head/loss/sampler/optimizer/schedule/AMP/EMA/256patches. All six epochs and source-only model/calibration locks were checked. Independent local recomputation reproduced raw/calibrated legacy and mean-image-F1 metrics, calibration-input hashes and available source-lock files. If weights were not downloaded, their server-side checksum lock remains explicitly marked remote-only in INDEPENDENT_VERIFICATION.json. Pixel independence does not prove identity/near-duplicate independence. Checkpoints/logs remain in private Kaggle outputs and local outputs/lodo where retrieved; large weights and sample-level arrays are not published in Git.
