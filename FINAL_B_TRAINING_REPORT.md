# Final Candidate B training and verification

Candidate B completed all eight authorized epochs. Epoch 8 was selected using raw DINO source-selection Challenge_Avg only. Epochs 7 and 8 satisfied the predeclared source-only extension rule. No further training was launched. The original LODO advancement gate remains failed; this final run was separately authorized as exploratory competition work.

## Training evidence

All 97,669 official training rows were available: PA100K 79,001, PETA 8,668 and Market1501 10,000. Source-selection and calibration contain 16,744 and 16,661 rows. The two validation pixels duplicated in training remain excluded. Original annotation identities, valid masks, domains, 40-attribute order, split hashes and pixel disjointness were independently checked against the sealed Candidate A partitions.

The original DINOv3 pretrained backbone and A epoch-30 head initialized the run. Only blocks 10/11, final normalization and the head trained. The square-root-domain sampler, masked balanced BCE, source-only class frequencies, learning rates, seed and preprocessing were retained. No all-domain SigLIP weights initialized DINO. The original SigLIP checkpoint is used unchanged only for the final probability ensemble.

| Epoch | Raw DINO source-selection Challenge_Avg | Training loss | Training plus selection seconds | AMP skips |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.821165073 | 0.204001 | 586.6 | 0 |
| 2 | 0.826920561 | 0.188596 | 600.1 | 0 |
| 3 | 0.831443647 | 0.181012 | 600.5 | 0 |
| 4 | 0.835234294 | 0.174066 | 603.0 | 1 |
| 5 | 0.835489379 | 0.167978 | 598.1 | 0 |
| 6 | 0.838134446 | 0.162896 | 599.7 | 0 |
| 7 | 0.839221383 | 0.158345 | 602.9 | 1 |
| 8 | 0.840004733 | 0.154447 | 602.3 | 0 |

The complete script took 5,026.07 seconds (83.77 minutes). The source backward probe measured 198.68 images/second and 0.763 GiB peak allocated memory. There were 12,216 attempted and 12,214 successful optimizer updates, with two AMP skips. Eight epochs consumed 781,352 draws: PA100K 462,524, PETA 153,861 and Market1501 164,967. Cumulative unique images were 78,749, 8,668 and 10,000 respectively; replacement sampling did not visit every PA100K training image.

## Source-only calibration and result

The new 40-threshold vector was independently refitted on the source-calibration partition only. The ensemble is 0.75 times original SigLIP raw sigmoid probabilities plus 0.25 times B raw sigmoid probabilities. Individually calibrated or thresholded branch outputs are never blended.

| Source-selection metric | Candidate B ensemble |
| --- | ---: |
| Challenge_Avg | 0.869305241 |
| mA | 0.860691698 |
| Instance-F1, harmonic of mean precision/recall | 0.878215200 |
| Precision | 0.853342272 |
| Recall | 0.904728941 |
| Mean-image F1 | 0.874474110 |

Candidate A scored 0.867806417 on the same source-selection rows. These are development source results, not independent LODO results or predicted official scores. The previously reported B LODO ensemble macro 0.802507754 and worst 0.781355654 remain the transfer evidence. B missed its historical +0.002 advancement margin versus A by approximately 0.000456; that decision has not been changed.

## Checkpoint and deployment correctness

Best checkpoint SHA256: `6e9fab596868099b1b8915349b5c9e935693c3a56836ea229c94037af7761c20`. Its 134 frozen backbone tensors exactly equal the verified original foundation. The deployment DINO asset SHA256 is `9752b481f4033fad78fd1c1e84163de6423d72730d3afcac5ebd38042b4b6873`.

Independent timm inference on one real image from each domain reproduced preprocessing and all 205 tokens exactly; final ensemble probabilities differed by at most 1.1921e-7. CLS, four registers and 200 spatial patches are handled explicitly. B preserves FP32 descriptor pooling used in fine-tuning; A's separate FP16 frozen-cache path remains unchanged. Five focused wrapper tests passed. The offline Python 3.11 / torch 2.4.1 CPU smoke passed with network and external model libraries disabled. CPU versus saved GPU AMP predictions differed by up to 0.006784, within the declared cross-precision tolerance; this is not a zero-difference GPU parity claim.

## Outstanding runtime gate

The exact-payload T4 surrogate hit its 300-second complete-workload timeout. Its last saved progress was 26,624 of 29,248 images at 296.717 seconds. Source archive staging passed in 9.115 seconds; all inference-file hashes matched the sealed local payload. This is an incomplete workload, not a measured full-run time. A linear extrapolation is approximately 326 seconds and is only an estimate. The T4 deployment timing gate failed; official A10 timing remains unverified. No B upload was performed. Fresh primary quota is 6.06 GPU hours remaining (23.94 used of 30).

The sealed ZIP is `UPAR2027_Candidate_B_Finetuned_DINOv3_75_25_v1.zip`, 650,813,407 bytes, with 18 whitelisted entries and passed CRC checks. SHA256: `f4eee47a694e113a7bb7c527f91592b7ff54fd8cbfdf85f70bb699014cf5f086`. It is stored privately under `D:\UPAR2027\diagnostics\competitive-final`. It excludes training images, feature caches, optimizer states and prediction arrays. Original metadata and required license notices are included.

Original submission 1003877, its ZIP/checkpoints/calibration and Candidate A remain preserved. Candidate C remains rejected. The source deadline record is October 22 at 12:00 UTC / 17:00 PKT, with the preferred upload target 13:00 PKT. Any official entry must account for Force_Last and the actual remaining submission allowance.
