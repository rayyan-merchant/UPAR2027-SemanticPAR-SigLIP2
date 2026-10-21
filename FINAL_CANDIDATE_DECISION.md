# Final decision: preserve submission 1003877

The frozen DINOv3 ensemble is useful, but **does not meet the predeclared promotion gate**. Do not train or upload an all-domain candidate from this frozen recipe.

| Held-out domain | SigLIP calibrated | DINO calibrated | 75/25 calibrated | Ensemble delta |
| --- | --- | --- | --- | --- |
| PA100k | 0.797104 | 0.754206 | 0.802375 | 0.005271 |
| PETA | 0.777228 | 0.724861 | 0.780285 | 0.003057 |
| Market1501 | 0.817448 | 0.783061 | 0.820231 | 0.002783 |
| Macro | 0.797260 | 0.754043 | 0.800964 | 0.003704 |
| Worst | 0.777228 | 0.724861 | 0.780285 | 0.003057 |

The fixed 75% SigLIP2 / 25% DINO ensemble improves every domain. Its macro score, 0.800964, falls 0.001336 short of the required 0.8023. Worst-domain and per-domain regression criteria pass. Standalone DINO is substantially weaker, with macro 0.754043 and worst 0.724861. These are development scores on previously inspected domains; they do not guarantee improving the official accepted score of 0.770450 or reaching 0.8021.

## Recommended next experiment

Review exactly one conservative fine-tune of the final two DINOv3 Transformer blocks. Keep all earlier blocks frozen, the original square-root sampler and class-balanced masked BCE, the same partitions, and the fixed ensemble weight. Use backbone LR about 5e-6 and head LR about 5e-5. Predeclare the source schedule and retain the existing promotion rule before evaluating targets. Preserve the frozen model as a fallback. No fine-tune has been launched.

This is worth considering because the frozen branch already improves all three target domains and its probability-residual correlations with SigLIP are only 0.747-0.776. Most of the gain comes from instance-F1, precision and recall; mA improves little overall and falls slightly on PETA. Clothing-color performance remains mixed. Modest feature adaptation could help, but that is a hypothesis, not a demonstrated remedy.

The evidence does not establish pretrained representation degradation, nor does it show that calibration or missing labels alone explain the gap. Every source attribute has positive and negative examples. Severe imbalance, four PA-held lower-orange positives, domain shift, frozen-feature limitations and unequal source sizes/updates remain confounded. The PA head is still best at its 30-epoch limit, which also limits conclusions about the architecture.

## Cost and resource limits

Fresh Kaggle quota: **10.66 GPU hours remaining**, 19.34 used. About 0.31 displayed hours have been spent across the canceled DINOv2 job, DINOv3 gate, failed first attempt and completed retry.

A single threefold experiment with two trainable blocks is tentatively estimated at 1-2 GPU-session hours. This has not been measured: first run a short source-only backward-throughput and memory gate. To stay inside the existing cumulative two-hour diagnostic allocation, cap further diagnostic work at 1.5 hours and stop if the measured projection cannot fit. Do not extend the cap or silently omit a fold.

If the resulting recipe is promoted, budget another approximately 1-2 hours for one all-domain DINO training run and 0.5-1 hour for exact deployment and packaging checks. These are planning estimates, not measured training times. Reserve at least six GPU hours until actual backward throughput gives a firmer schedule. Cached-head timings of 24-86 seconds cannot be extrapolated to backbone fine-tuning.

## Submission schedule

- **October 21, by 18:00 PKT:** review this result and decide whether to authorize the one limited fine-tune. Include the verified frozen comparison and non-promotion decision in academic reporting today. Academic portal details, human/team fields and acceptance remain unverified.
- **October 21, by 22:00 PKT:** freeze a final recipe only if its complete threefold comparison passes the unchanged promotion rule. Otherwise retain 1003877 and stop model experiments.
- **October 22, by 06:00 PKT:** complete any conditionally promoted all-domain training and permitted validation calibration. Preserve the original SigLIP weights and calibration.
- **October 22, 06:00-10:00 PKT:** package offline inference with original metadata, exact 40-attribute ordering and license notices. Verify real-image predictions, network-disabled loading, and the complete decode/both-forward-passes/calibration/serialization path in at most 240 seconds. Reject a candidate that misses that margin.
- **October 22, 10:00-12:00 PKT:** deliver the concrete package, hashes and evidence for explicit user upload approval. Preferred upload is **13:00 PKT**; hard cutoff is **17:00 PKT**. No automatic upload or organizer email.

The frozen diagnostic is complete and independently audited. All 31 locked baseline report files and the accepted ZIP, best/last checkpoints and calibration were verified unchanged. No GPU job is active. Follow-up monitoring stops after this report; the next experiment awaits review of this evidence and resource plan.
