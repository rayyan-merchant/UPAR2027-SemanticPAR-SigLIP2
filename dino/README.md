# DINOv3 complementary branch

This directory contains the complementary visual representation, its regional attribute head, strict source-only experiments and separate A/B deployment wrappers. It is not an officially scored replacement for submission 1003877. [Architecture](../DINO_ARCHITECTURE.md) · [Frozen results](../DINO_LODO_RESULTS.md) · [B LODO](../DINO_FINETUNE_LODO.md)

| Code | Responsibility |
|---|---|
| [v3_model.py](v3_model.py) | Torch-only DINOv3 ViT-B/16 matching the verified timm convention, including registers and RoPE |
| [model.py](model.py) | Shared six-descriptor regional pooling/head components, plus preserved DINOv2 fallback implementation |
| [reference.py](reference.py) and [v3_feasibility.py](v3_feasibility.py) | Reference parity and measured source-only feasibility |
| [v3_experiment.py](v3_experiment.py) / [v3_runner.py](v3_runner.py) | Compact frozen descriptor extraction and three-fold head evaluation |
| [v3_finetune.py](v3_finetune.py) / [finetune_runner.py](finetune_runner.py) | Candidate B final-two-block LODO adaptation |
| [final_a.py](final_a.py) | Exploratory final all-domain frozen head |
| [final_b.py](final_b.py) / [final_b_runner.py](final_b_runner.py) | Exploratory all-domain B training and source-only checkpoint/calibration locks |
| [ensemble_submission.py](ensemble_submission.py) | Candidate A raw 75/25 probability mixture and new global calibration |
| [ensemble_finetuned_submission.py](ensemble_finetuned_submission.py) | Candidate B wrapper preserving the fine-tuning FP32 descriptor path |
| [v3_api.py](v3_api.py) | Native DINOv3 inference support |
| [licenses/](licenses/) | Meta DINOv3, timm and retained upstream notices |

The canvas is 320x160. CLS plus four registers precede 200 spatial tokens. The head attends over CLS, spatial mean, overlapping head/upper/lower means and spatial maximum. LODO folds use their own source-trained frozen heads; they never use A's all-domain head. Only the separately authorized final all-domain B run initializes from A's all-domain head.

The earlier official DINOv2 fallback was cancelled when the legitimate timm DINOv3 route passed access and correctness gates. It has no completed comparative score. Its code/notices remain for provenance. Do not confuse the fallback implementation with the active DINOv3 backbone.

Both A/B wrappers blend raw branch probabilities before source-only ensemble calibration. They do not blend thresholded or individually calibrated outputs. A/B trained weights remain private and are not included here. See [final B acceptance](../CANDIDATE_B_ACCEPTANCE.md) for its correctness pass and unresolved runtime gate.
