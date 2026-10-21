# Frozen DINOv3 LODO: complete, independently verified

Private Kaggle upar2027-dinov3-frozen-threefold version2 COMPLETE. All19Kaggle tests passed. Fresh general-purpose DINOv3 pretrained backbone frozen; exact three existing disjoint source-training/selection/calibration/target partitions, original square-root-domain sampler and source-only class-balanced masked BCE. All three heads ran30epochs maximum with source-only early stopping/checkpoint selection. Target inference followed saved checkpoint/config/calibration/split locks; no target-driven settings were changed. No all-domain model or competition submission was created.

## Held-out scores

| Setting | PA100K | PETA | Market1501 | Macro | Worst |
| --- | --- | --- | --- | --- | --- |
| siglip_raw | 0.786128 | 0.769342 | 0.813343 | 0.789605 | 0.769342 |
| siglip_calibrated | 0.797104 | 0.777228 | 0.817448 | 0.797260 | 0.777228 |
| dino_raw | 0.742350 | 0.714441 | 0.775750 | 0.744181 | 0.714441 |
| dino_calibrated | 0.754206 | 0.724861 | 0.783061 | 0.754043 | 0.724861 |
| ensemble_raw | 0.787468 | 0.770556 | 0.814629 | 0.790884 | 0.770556 |
| ensemble_calibrated | 0.802375 | 0.780285 | 0.820231 | 0.800964 | 0.780285 |

| Domain | Setting | mA | Instance-F1 | Precision | Recall | Label-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| PA100k | siglip_calibrated | 0.786334 | 0.808174 | 0.772357 | 0.847475 | 0.523229 |
| PA100k | dino_calibrated | 0.717711 | 0.794613 | 0.768303 | 0.822788 | 0.443768 |
| PA100k | ensemble_calibrated | 0.787098 | 0.818257 | 0.787368 | 0.851668 | 0.533352 |
| PETA | siglip_calibrated | 0.769719 | 0.784886 | 0.767434 | 0.803150 | 0.537806 |
| PETA | dino_calibrated | 0.700299 | 0.751208 | 0.724008 | 0.780531 | 0.439011 |
| PETA | ensemble_calibrated | 0.769171 | 0.791726 | 0.772246 | 0.812214 | 0.546747 |
| Market1501 | siglip_calibrated | 0.800970 | 0.834618 | 0.821221 | 0.848460 | 0.527449 |
| Market1501 | dino_calibrated | 0.760447 | 0.807061 | 0.774224 | 0.842806 | 0.459721 |
| Market1501 | ensemble_calibrated | 0.802833 | 0.838399 | 0.824107 | 0.853196 | 0.528698 |

Challenge_Avg is the arithmetic domain macro of the per-domain harmonic mean of mA and instance-F1. Primary instance-F1 follows the locked baseline convention: harmonic mean of mean image precision and mean image recall. Both-supported labels enter mA; exact private organizer edge cases remain unresolved. These previously inspected targets are development evaluation, not untouched certification of generalization or a prediction of the official leaderboard score.

## Selection, calibration and gaps

| Held out | Selected epoch / run epochs | DINO source raw | DINO source calibrated | DINO target calibrated | Source-target gap | Ensemble source calibrated | Ensemble source-target gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 30 / 30 | 0.783827 | 0.800012 | 0.754206 | 0.045805 | 0.852052 | 0.049677 |
| PETA | 27 / 30 | 0.808107 | 0.825068 | 0.724861 | 0.100207 | 0.864810 | 0.084525 |
| Market1501 | 25 / 30 | 0.795283 | 0.811239 | 0.783061 | 0.028178 | 0.870082 | 0.049851 |

Source-selection scores are descriptive on the selection partition used for checkpoint selection. Calibrated source-selection metrics use thresholds fit only on its disjoint source-calibration partition; no target fitting. Ensemble source-selection metrics use the corresponding preserved SigLIP source predictions. Target gaps confound domain shift with class balance, source domain composition and training-set size; they do not isolate a causal mechanism.

| Held out | DINO raw | DINO calibrated | Transfer effect | Ensemble raw | Ensemble calibrated | Transfer effect |
| --- | --- | --- | --- | --- | --- | --- |
| PA100k | 0.742350 | 0.754206 | 0.011856 | 0.787468 | 0.802375 | 0.014908 |
| PETA | 0.714441 | 0.724861 | 0.010419 | 0.770556 | 0.780285 | 0.009730 |
| Market1501 | 0.775750 | 0.783061 | 0.007311 | 0.814629 | 0.820231 | 0.005601 |

Source-only calibration improved Challenge_Avg for both branches in all three targets, mainly trading precision for recall and label-balanced accuracy. DINO alone remains substantially below SigLIP2. Calibration does not close the entire standalone gap.

## Source convergence and compute

| Held out | Source train rows | Images consumed | Attempted / successful updates | AMP skips | Head wall seconds | Final unique source images |
| --- | --- | --- | --- | --- | --- | --- |
| PA100k | 18668 | 560040 | 4380 / 4380 | 0 | 24.32 | 18668 |
| PETA | 89001 | 2670030 | 20880 / 20876 | 4 | 85.78 | 89001 |
| Market1501 | 87669 | 2630070 | 20550 / 20546 | 4 | 78.99 | 87669 |

PA is best at the30epoch limit and still improving; PETA peaks at27,Market at25. Do not claim all three converged or compare raw epoch numbers with the six-epoch fine-tuned SigLIP baseline. PA source size18,668 is much smaller than PETA89,001 or Market87,669. Actual attempted/successful update counts and8total AMP-skipped updates are retained, not assumed zero. Skipped updates still consumed their sampled source images. Source losses were finite. Epoch curves, source scores, draws and unique counts are retained in SOURCE_CURVES_SUPPORT.json and SOURCE_CURVES.csv; the numerical trajectories are directly reusable for plotting.

All source images were eventually seen at least once in each fold. Measured exposure fractions match the expected square-root sampler: PA-held Market/PETA .51852/.48148; PETA-held Market/PA .26297/.73703; Market-held PA/PETA .75100/.24900. Frequencies/BCE weights were computed from the actual source partition, not sampled or target data. Every one of40source attributes has positive and negative support in all3folds. PA-held lower-orange has only4source positives; this is sparse support, not missing supervision. Full source counts and target TP/TN/FP/FN are in the aggregate CSV/JSON files.

## Attribute behavior and uncertainty

| Group | PA mA delta | PETA mA delta | Market mA delta | PA label-F1 delta | PETA label-F1 delta | Market label-F1 delta |
| --- | --- | --- | --- | --- | --- | --- |
| age | 0.005294 | 0.013100 | 0.000465 | 0.016656 | 0.011922 | 0.009717 |
| upper_colors | 0.006659 | -0.003993 | -0.000181 | 0.017182 | 0.007108 | -0.001743 |
| lower_colors | -0.012164 | -0.009973 | 0.000170 | -0.000075 | 0.004952 | 0.001696 |
| accessories | 0.003930 | 0.013017 | 0.005300 | 0.012425 | 0.018392 | -0.010898 |

Group deltas compare calibrated ensemble with calibrated SigLIP. Age and accessory mA rise across all three domains; upper/lower clothing colors are mixed, including lower-color mA regressions on PA/PETA. Accessory label-F1 falls onMarket despite higher group mA. There is no uniformly improved attribute group. Largest supported-label changes below include low-support cases and should not be interpreted without the support columns.

| Domain | Attribute | mA delta | Label-F1 delta | Target positives | Source positives |
| --- | --- | --- | --- | --- | --- |
| PA100k | UpperBody-Color-Brown | 0.043320 | 0.001182 | 171 | 625 |
| Market1501 | Accessory-Hat | 0.032310 | 0.005818 | 480 | 4375 |
| PA100k | UpperBody-Color-Other | 0.030023 | 0.066305 | 1053 | 956 |
| Market1501 | Age-Young | 0.022778 | -0.004043 | 39 | 4389 |
| PETA | Accessory-Glasses-Normal | 0.020884 | 0.033429 | 499 | 14210 |
| PETA | LowerBody-Type-Trousers&Shorts | 0.018094 | 0.000489 | 6562 | 78291 |
| PA100k | LowerBody-Color-White | 0.016511 | 0.007287 | 205 | 1314 |
| PETA | Age-Adult | 0.016352 | 0.001731 | 6287 | 83452 |
| PETA | LowerBody-Color-Orange | -0.099784 | -0.015873 | 5 | 255 |
| PA100k | LowerBody-Color-Red | -0.091247 | -0.096683 | 87 | 62 |
| PA100k | LowerBody-Color-Yellow | -0.041265 | 0.000000 | 24 | 168 |
| PETA | UpperBody-Color-Brown | -0.032202 | 0.037590 | 493 | 2507 |
| Market1501 | Accessory-Glasses-Normal | -0.028296 | -0.096738 | 2855 | 13038 |
| PA100k | LowerBody-Color-Brown | -0.025378 | 0.014131 | 88 | 1456 |
| PA100k | Accessory-Glasses-Normal | -0.023261 | 0.011127 | 1302 | 2404 |
| PETA | LowerBody-Color-Red | -0.022822 | 0.009269 | 43 | 708 |

Full per-attribute and per-group raw/calibrated tables: reports/dino_v3/results/ATTRIBUTE_METRICS.csv and GROUP_METRICS.csv. Zero-target-positive labels and very rare labels have unstable/undefined balanced accuracy; do not turn these observations into hidden-test assumptions.

## Scorer sensitivity

| Setting | Mean-image-F1 Challenge macro | Mean-image-F1 worst | All-valid safe-zero mA + mean-image-F1 macro |
| --- | --- | --- | --- |
| siglip_calibrated | 0.794709 | 0.774298 | 0.786422 |
| dino_calibrated | 0.750901 | 0.721008 | 0.743518 |
| ensemble_calibrated | 0.798470 | 0.777373 | 0.790094 |

The ensemble improvement remains positive under both instance-F1 conventions and the all-valid safe-zero sensitivity. Numerical promotion criteria were predeclared on the locked primary convention and are not moved. Mean-image sensitivity is diagnostic, not a substitute selected after seeing results.

## Verification boundary

Independent CPU confusion-count calculations reproduce all recorded target metrics. Source-only DINO and ensemble thresholds refit exactly. Source-selection metrics, row identities, all CSV/checkpoint/config/calibration locks, original foundation/code identity, source prevalence/BCE weights, source exclusion and exposure accounting verify. The three small selected heads were downloaded privately toD: and hash-checked. The1.208GBcache remains private onKaggle; its recorded identity/shape/hash were verified, but its payload was not downloaded and independently rehashed locally. No claim of a second full-cache integrity pass is made. Original31baseline report files and accepted ZIP/best/last/calibration hashes remain unchanged.
