# Setup and reproduction

## Review without data or a GPU

From the repository root, run:

```bash
python scripts/verify_assignment_results.py
```

This standard-library-only check validates the 73 published CSV rows, official score, fold macro/worst arithmetic and decision margins. It does not re-score raw private predictions. The scientific methods and measured values are summarized in the report, results table and `configs/experiments.json`.

## CPU correctness tests

Use Python 3.12, torch 2.4.1 CPU and torchvision 0.19.1, then install `requirements-dino.txt` and run `python -m pytest -q tests`. The root README has complete installation commands. Tests use generated fixtures and small tensors; no cloud job or competition entry is created. `assignment/LOCAL_VERIFICATION.json` records the latest verified run.

## Training with authorized inputs

Obtain official UPAR Track 1 images/annotations and the official PETA filename mapping under their original terms. Keep originals read-only. `python -m upar.inspect_data --help` shows the data-audit arguments; `python -m upar.train --help` shows training arguments and required data/parity/feasibility gates. Settings used by the research are in `configs/experiments.json` and `configs/training.json`.

Use `scripts/prepare_lodo_diagnosis.py` to prepare source-training, source-selection, source-calibration and held-out partitions. It requires the authorized annotations and audited pixel manifest. The target domain must not enter training, source class frequencies, selection or calibration. Never initialize a LODO fold from an all-domain PAR checkpoint.

The original model is in `upar/model.py`; its masked balanced loss and square-root sampler are in `upar/train.py` and `upar/data.py`. DINOv3 and the six-descriptor head are in `dino/v3_model.py` and `dino/model.py`. Frozen heads use `dino/v3_experiment.py`; partial adaptation uses `dino/v3_finetune.py`. Final all-domain recipes are in `dino/final_a.py` and `dino/final_b.py`. Candidate C's fixed augmentation is in `upar/robustness.py`.

For repeated folds, stage verified archives once using `kaggle/archive_stage.py`. Dataset-specific cloud runners require matching private input mounts, prepared experiment plans and hash-pinned source artifacts. Those operational inputs are intentionally not published; the runners are execution source, not a one-command recreation of the authors' private workspace. Internal launch/build/account helpers are excluded. CPU tests and the published metric table can be reviewed without them.

## Foundations and inference

Use the official Google SigLIP2-Base-NaFlex foundation and the maintainer-supported timm DINOv3 ViT-B/16 checkpoint, not PAR-fine-tuned substitutes. The DINOv3 safetensors SHA256 is `1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`. Follow original licenses and perform reference/native checks before optimization.

The original inference API is `submission_template/run.py`; A uses `dino/ensemble_submission.py`, and B uses `dino/ensemble_finetuned_submission.py`. Both ensembles blend raw probabilities before a new source-only threshold vector. B preserves its fine-tuning FP32 descriptor path rather than A's frozen-cache FP16 round-trip. Actual trained checkpoints/calibration are not redistributed, so image-level prediction requires the matching authorized model artifacts.

## Rebuild the report

Install `requirements-report.txt`, then run `python scripts/build_assignment_report.py`. The editable source is `assignment/report_content.json`; the report uses a flexible page count so equations, tables and interpretations remain readable. Internal lab notes, detailed execution logs and cloud receipts are retained outside the submitted repository.

The PDF builder embeds a Unicode font. It uses Segoe UI on Windows or DejaVu Sans on Linux; installing the optional `matplotlib` package also supplies DejaVu Sans. The PDF and matching Markdown are generated from the same structured source.
