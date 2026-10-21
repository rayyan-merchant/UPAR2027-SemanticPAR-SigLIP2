# Candidate C: surveillance-degradation augmentation

All three six-epoch LODO folds completed. Independent metric reproduction, source-only calibration refits, original split hashes and available source/configuration locks passed. Checkpoint bytes remain privately on Kaggle and were verified by the remote evaluation, not rehashed locally.

**Do not advance this recipe.** Calibrated macro is 0.798076 (+0.000816 versus original SigLIP), while worst-domain PETA falls to 0.776418 (-0.000810). The predeclared +0.003 macro and worst-domain margins both fail. No all-domain C training is launched.

The original architecture, foundation, sampler, class frequencies/BCE, optimizer, upper-four-block schedule, EMA and source-selection/calibration procedure are unchanged. The isolated augmentation policy preserves the existing mirror/crop/contrast/erasure geometry and adds bounded down/up sampling (p=.30, scale .65-1), gamma (p=.20, .9-1.1), brightness (p=.5, .9-1.1), JPEG (p=.30, quality 60-95) and blur (p=.15, sigma .2-.7). No external PAR model or held-out training labels are used.

| Held out | Raw Challenge | Calibrated Challenge | mA | Instance-F1 | Precision | Recall | Source score | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PA100k | 0.787039 | 0.797197 | 0.786977 | 0.807685 | 0.773603 | 0.844909 | 0.840438 | 6 |
| PETA | 0.768017 | 0.776418 | 0.770637 | 0.782287 | 0.760553 | 0.805299 | 0.846913 | 5 |
| Market1501 | 0.813845 | 0.820613 | 0.803434 | 0.838543 | 0.826087 | 0.851379 | 0.857571 | 6 |

## Attribute groups and convergence

| Held out | Age mA | Upper colors mA | Lower colors mA | Accessories mA | Updates attempted / successful | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| PA100k | 0.676640 | 0.817364 | 0.760984 | 0.736572 | 1752 / 1752 | 1384.7 |
| PETA | 0.667725 | 0.795208 | 0.773600 | 0.702459 | 8346 / 8343 | 4102.3 |
| Market1501 | 0.780620 | 0.865670 | 0.750116 | 0.706401 | 8220 / 8217 | 4158.9 |

Suite time was 6002.41 seconds (1.667 hours), plus a 60-second debit for a bootstrap syntax failure before training. Source convergence diagnostics and all per-attribute support/error counts are retained in reports/competitive_final/candidate_c. Six epochs are a fixed comparison budget, not proof of full convergence. Different source sizes, sparse labels and one seed limit causal attribution. The policy gives mixed transfer rather than evidence of a consistent robustness improvement; source-validation scores cannot override the negative worst-domain result. Both harmonic-of-mean-P/R and mean-image-F1 conventions remain reported because the exact organizer implementation is private.
