# SigLIP2 implementation guide

This is the original officially evaluated branch. The model uses SigLIP2-Base-NaFlex, semantic attribute queries and masked attention over image patches, with a global residual. Original accepted submission: **1003877**, Challenge_Avg **0.770450**. [Task](../assignment/TASK_AND_APPROACH.md) · [Results](../TRAINING_RESULTS.md)

| Code | Responsibility |
|---|---|
| [model.py](model.py) | Native visual backbone, attribute queries, cross-attention and classification |
| [attributes.py](attributes.py) | Required order of all 40 attributes |
| [preprocess.py](preprocess.py) | Aspect-aware NaFlex image-to-patch preprocessing |
| [data.py](data.py) | Label masks, source-only frequencies and domain sampling |
| [train.py](train.py) | Masked weighted BCE, warmup, partial backbone adaptation, EMA and resumable training |
| [metrics.py](metrics.py) | Local metric conventions and threshold selection |
| [calibrate.py](calibrate.py) | Global 40-attribute calibration |
| [parity.py](parity.py) | Reference/native pretrained checks before training |
| [feasibility.py](feasibility.py) | Actual GPU forward/backward feasibility gate |
| [lodo.py](lodo.py) | Source/target partition isolation |
| [robustness.py](robustness.py) and [robustness_train.py](robustness_train.py) | Candidate C's isolated degradation augmentation experiment |
| [package.py](package.py) | Minimal offline competition archive assembly |
| [../submission_template/run.py](../submission_template/run.py) | Official inference API wrapper using the original metadata contract |

From the repository root, `python -m upar.train --help` shows required private inputs and gate reports. CPU tests run without training data; reproducing measured accuracy requires authorized images and the hash-matching trained weights. Read [REPRODUCING.md](../docs/REPRODUCING.md) before running training. Generic CLI defaults do not supersede the [executed original configuration](../reports/kaggle_v4/training/training_config.json).
