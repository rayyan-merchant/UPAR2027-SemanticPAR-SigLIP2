# Candidate B: partial DINOv3 fine-tuning

All three strict LODO folds completed six epochs. Independent metric recomputation, source-only threshold refits, original split hashes, label/mask identity and available source lock files passed. Final checkpoint payloads remain private on Kaggle and were not independently rehashed locally.

The fixed 75/25 ensemble improves frozen-ensemble macro from 0.800964 to **0.802508**, and worst domain from 0.780285 to **0.781356**. Macro mA increases from 0.786367 to **0.788772**. This is promising, but the +0.001544 macro gain falls below the predeclared +0.002 advancement margin (required 0.802964). No final all-domain B training is launched. The earlier absolute 0.8023 gate is not substituted for this branch-specific margin.

Only DINO blocks 10/11, final normalization and the source-trained frozen head were trainable. Each fold uses the original foundation and its own source-only frozen head, not Candidate A. Backbone/head AdamW rates are 5e-6/5e-5, masked balanced BCE, original square-root-domain sampler, seed 2027, aspect-fit 320x160 and horizontal flip probability 0.5. Six epochs use source-only selection/calibration and a locked target evaluation. SigLIP fold predictions and 75/25 raw-probability mixing stay fixed.

| Held out | DINO raw | DINO calibrated | Ensemble raw | Ensemble calibrated | mA | Instance-F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PA100k | 0.771822 | 0.780643 | 0.789246 | 0.803448 | 0.790330 | 0.817009 | 0.787075 | 0.849310 |
| PETA | 0.743945 | 0.752457 | 0.772123 | 0.781356 | 0.770384 | 0.792644 | 0.775098 | 0.811003 |
| Market1501 | 0.811044 | 0.817344 | 0.817050 | 0.822720 | 0.805603 | 0.840580 | 0.822637 | 0.859322 |

## Source convergence and compute

| Held out | Source-selected epoch | Source score | First / final source score | Attempted / successful updates | AMP skips | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| PA100k | 6 | 0.822169 | 0.802624 / 0.822169 | 1752 / 1752 | 0 | 948.0 |
| PETA | 5 | 0.841636 | 0.830624 / 0.840224 | 8346 / 8346 | 0 | 3266.0 |
| Market1501 | 5 | 0.836946 | 0.818839 / 0.836252 | 8220 / 8220 | 0 | 2988.6 |

The suite runner took 4034.03 seconds (1.121 hours), plus the earlier failed bootstrap debit of 120 seconds. Concurrent source backward throughput was conservatively 196.23 images/s. All 40 source attributes have positives and negatives; sparse support remains important (PA-held lower-orange has only four training positives). Different source sizes and update counts confound comparisons between held-out domains. One seed and previously inspected development targets do not establish statistical significance or unbiased certification.

## Calibrated ensemble attribute groups

| Held out | Age mA | Upper colors mA | Lower colors mA | Accessories mA |
|---|---:|---:|---:|---:|
| PA100k | 0.697804 | 0.823459 | 0.752103 | 0.749450 |
| PETA | 0.683918 | 0.796160 | 0.763185 | 0.711536 |
| Market1501 | 0.780793 | 0.870436 | 0.749738 | 0.713047 |

Complete raw/calibrated scores, alternative mean-image-F1 and unsupported-label conventions, group metrics and source exposure curves are retained in reports/competitive_final/candidate_b. The exact organizer scorer remains private. The measured mA gain supports further scientific interest, but does not justify bypassing the declared advancement rule or guarantee an official improvement.
