# Fixed75/25 SigLIP2 + frozen DINOv3: useful but not promoted

| Held-out domain | SigLIP calibrated | DINO calibrated | 75/25 calibrated | Ensemble delta |
| --- | --- | --- | --- | --- |
| PA100k | 0.797104 | 0.754206 | 0.802375 | 0.005271 |
| PETA | 0.777228 | 0.724861 | 0.780285 | 0.003057 |
| Market1501 | 0.817448 | 0.783061 | 0.820231 | 0.002783 |
| Macro | 0.797260 | 0.754043 | 0.800964 | 0.003704 |
| Worst | 0.777228 | 0.724861 | 0.780285 | 0.003057 |

Use the predeclared weighted average of raw sigmoid probabilities, not logits or already calibrated outputs. A separate global40threshold vector was fit on each fold's source-calibration data only, then locked before target inference. SigLIP checkpoints/predictions were unchanged. No ensemble-weight or threshold search used targets.

The ensemble improves macro by0.003704 and worst-domain by0.003057. All three domains improve, but macro0.800964 falls0.001336 short of the fixed0.8023criterion. Worst0.780285>=0.7772 and no-regression criteria pass. **The frozen ensemble is not promoted.**

## What changes

Macro mA0.785674->0.786367 (+0.000693),instance-F10.809226->0.816127 (+0.006901),precision0.787004->0.794574 (+0.007570),recall0.833028->0.839026 (+0.005998),label-F10.529495->0.536266. Most aggregate benefit is instance-level precision/recall, not a broad recovery of label-balanced mA.

PA:mA+0.000764,F1+0.010083;PETA:mA-0.000548,F1+0.006840;Market:mA+0.001864,F1+0.003781. PETA's Challenge gain should not be described as a balanced-accuracy gain.

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

| Group | PA mA delta | PETA mA delta | Market mA delta | PA label-F1 delta | PETA label-F1 delta | Market label-F1 delta |
| --- | --- | --- | --- | --- | --- | --- |
| age | 0.005294 | 0.013100 | 0.000465 | 0.016656 | 0.011922 | 0.009717 |
| upper_colors | 0.006659 | -0.003993 | -0.000181 | 0.017182 | 0.007108 | -0.001743 |
| lower_colors | -0.012164 | -0.009973 | 0.000170 | -0.000075 | 0.004952 | 0.001696 |
| accessories | 0.003930 | 0.013017 | 0.005300 | 0.012425 | 0.018392 | -0.010898 |

## Complementarity

| Held out | Probability-residual correlation | Binary-error correlation at0.5 | Decision disagreement | Double fault |
| --- | --- | --- | --- | --- |
| PA100k | 0.770767 | 0.536641 | 0.055610 | 0.036313 |
| PETA | 0.775625 | 0.547614 | 0.061999 | 0.042737 |
| Market1501 | 0.747056 | 0.558203 | 0.055330 | 0.039235 |

Residual correlation0.747-0.776 and binary-error correlation0.537-0.558 show useful, incomplete agreement. Correlation depends on label prevalence and is descriptive; it is not proof of causality or independent representations. Consistent target gains support complementarity despite the standalone gap, but a single development seed/previously inspected targets do not certify future or official-score gains. Calibration transfer contributes substantially: raw ensemble macro0.790884 versus rawSigLIP0.789605; calibrated ensemble0.800964 versus calibratedSigLIP0.797260.

## Decision

Do not deploy DINO standalone or the current frozen ensemble. The predeclared review-only near-useful conditions are met (macro>=.8003,worst>=.7752,residual correlations<.85), which supports considering exactly one limited final-two-block experiment. It does not override the primary promotion gate or authorize an automatic final all-domain run. Retain original square-root sampling and masked class-balanced BCE; equal-domain sampling remains rejected by its completed controlled comparison. No further ensemble-weight search.

## Inference and packaging boundary

DINO-only warmed T4 projection for29248decode/preprocess/forward/pooling images is~102s; the previous scored SigLIP path was~110s on its official runtime. Adding these heterogeneous numbers is not an ensemble benchmark. Model loading, independent aspect preprocessing,40head logits,raw-probability mixing,newcalibration,serialization and the exact A10 deployment path still need an inclusive <=240s measurement. Never blend the accepted model's threshold-shifted output with raw DINO probabilities. Preserve accepted metadata and calibration; a future ensemble needs a separate wrapper/calibration file. No promoted/all-domain candidate or ZIP exists.
