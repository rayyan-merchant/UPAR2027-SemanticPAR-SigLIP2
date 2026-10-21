# DINOv3 LODO results — pending benchmark

No DINOv3 full descriptor extraction or head training has started. The bounded GPU benchmark must finish and its download/hash/parity/throughput/quota evidence be reported before full extraction. Strict original LODO split hashes, source-only frequency/sampling/selection/calibration and pre-target locks are unchanged. Original square-root-domain sampler retained. Reuse frozen six-descriptor40-head, AdamW3e-4/dropout.2/batch128/masked source-balanced BCE/seed2027, maximum30head epochs with source-only stopping. This is development evaluation on previously inspected domains, not an untouched unbiased test.

SigLIP2 calibrated baselines remain PA.797104355/PETA.777228370/Market.817447759,macro.797260162,worst.777228370. Do not replace missing DINO results with projections. Preserve score convention/zero-support sensitivities and source/target support/confounders.
