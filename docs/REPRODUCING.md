# Reproducing code checks and experimental results

## Reproduce the included results table

From the repository root, run `python scripts/verify_assignment_results.py`. It reads official receipts and aggregate fold evaluations, checks numeric values and failed decision gates, and reproduces `assignment/RESULTS.csv`. This needs only Python's standard library, no models/data/network. It validates the transcription and arithmetic of retained reports, not raw image inference. `assignment/RESULTS.json` identifies the source evidence for each record.

## Run CPU correctness checks

Install Python 3.11/3.12, torch 2.4.1 CPU and `requirements-dino.txt`, then `python -m pytest -q tests`. These are code/contract checks, not a claim of retrained accuracy or A10 timing. GPU/reference parity and environment receipts from actual jobs are retained in reports. To put pytest temporary files outside a full system drive, append `--basetemp <writable-path>`.

## Reproduce training with authorized private inputs

1. Obtain official UPAR Track 1 annotations/images and the official PETA mapping. Keep originals read-only. Run `python -m upar.inspect_data --help` and use the data-audit example in `docs/ORIGINAL_IMPLEMENTATION.md`.
2. Obtain Google SigLIP2-Base-NaFlex and the maintainer-supported timm DINOv3 ViT-B/16 foundation under their original terms. The DINO safetensors SHA256 is `1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`. Preserve provenance; do not substitute PAR-fine-tuned weights.
3. Execute the native/reference preprocessing and model parity checks. Training CLI: `python -m upar.train --help`. Required data/parity/feasibility reports are intentional gates; a code-test pass does not bypass them. The original executed recipe is `reports/kaggle_v4/training/training_config.json`.
4. For LODO, `scripts/prepare_lodo_diagnosis.py` builds disjoint source-training/selection/calibration and sealed-target manifests. `reports/lodo/EXPECTED_SPLITS.json` and later three-fold receipts identify the locked partitions. Do not use target labels for selection or thresholds. Full private split CSVs and sample-level arrays are not publicly distributed; recreating exact bytes requires the authorized annotations and recorded configuration/hash checks.
5. Stage verified domain archives once with `scripts/build_private_archives.py` and `kaggle/archive_stage.py`; use private cloud inputs. Do not restore the old per-image mounted-copy loop. Staging, I/O and pretraining watchdogs remain mandatory.
6. Baseline jobs are orchestrated by `kaggle/lodo_threefold_runner.py`. Frozen descriptors/heads use `dino/v3_runner.py` and `dino/v3_experiment.py`; B LODO uses `dino/finetune_runner.py` / `dino/v3_finetune.py`; C uses `kaggle/candidate_c_runner.py`. Their recorded plans/configurations and hashes, not generic CLI defaults, define the executed experiments. Re-running requires private input mounts, explicit quota/deadline review and a separately prepared cloud job.
7. Final A and B code is in `dino/final_a.py` and `dino/final_b.py` / `final_b_runner.py`. B LODO must start from each source-only frozen head; only the separately authorized all-domain final run starts from A's all-domain head. Final B completed eight epochs with source-only extensions; no further extension was run.

## Independently audit private prediction artifacts

The retained audit scripts include `scripts/summarize_lodo.py`, `scripts/compare_lodo_sampling.py`, `scripts/analyze_dinov3_frozen.py` and generalization analysis. They need the corresponding private NPZ/CSV/checkpoint artifacts, which are deliberately absent from this public source bundle. Some historical scripts use recorded workspace paths or write default report directories. Run them in a copy with explicit output paths; never overwrite locked baseline reports. Read the script arguments before execution. Public independent-verification receipts state which locks/checkpoint bytes were checked locally and which were verified only by the remote evaluator.

## Reproduce offline inference and benchmark

Original SigLIP packaging is `python -m upar.package --help`. A uses `dino/ensemble_submission.py`; B uses `dino/ensemble_finetuned_submission.py` because B's fine-tuned descriptor path preserves FP32 rather than A's cached-feature FP16 round-trip. Copy the chosen wrapper as root `run.py`, include the original metadata and matching private model/calibration assets, then block network and optional model-library imports during verification. This repository does not include deployable trained weights, so it cannot produce the measured score from code alone.

`scripts/benchmark_offline_ensemble.py` measures a cold-process representative workload using explicit permitted source image paths, without labels. The executed B probe added progress and disk-serialization instrumentation; its failed timeout and exact-payload receipt are retained in `reports/competitive_final/final_b_results/runtime_v1`. It processed 26,624 images by 296.717 seconds before a 300-second timeout. There is no completed B full-workload runtime or A10 certificate. The official original submission's 109.91-second recorded inference is separate evidence. Never promote a T4 projection to an official A10 pass.

## Evidence boundaries and privacy

One seed and repeatedly inspected development targets limit generalization claims. Both local F1 conventions and unsupported-label sensitivities remain available. Missing private weights/data is an intentional licensing/privacy boundary, not an implicit claim that reproduction is turnkey. No tests, results verifier or document builder launches cloud training or submits to CodaBench.
