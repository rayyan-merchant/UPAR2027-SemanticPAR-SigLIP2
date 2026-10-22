# Script entry points

Run these from the repository root. The first three commands need only Python's standard library, except that building the PDF uses `requirements-report.txt`. They do not allocate cloud compute.

| Purpose | Entry point |
|---|---|
| Verify saved assignment results and baseline hashes | `python scripts/verify_assignment_results.py` |
| Verify repository navigation and release contents | `python scripts/check_assignment_release.py` |
| Rebuild the three-page PDF | `python scripts/build_assignment_report.py` |
| Inspect allowed source data / preserve original paths | [prepare_kaggle_data.py](prepare_kaggle_data.py), [review_known_duplicates.py](review_known_duplicates.py) |
| Build private sequential-read data archives | [build_private_archives.py](build_private_archives.py) |
| Prepare strict source/target folds | [prepare_lodo_diagnosis.py](prepare_lodo_diagnosis.py) |
| Independently summarize baseline LODO private artifacts | [summarize_lodo.py](summarize_lodo.py) |
| Audit the single equal-sampling comparison | [compare_lodo_sampling.py](compare_lodo_sampling.py) |
| Audit frozen DINO and fixed ensemble private artifacts | [analyze_dinov3_frozen.py](analyze_dinov3_frozen.py) |
| Investigate the original validation/official gap | [generalization_analysis.py](generalization_analysis.py) |
| Check trained offline submission behavior | [verify_trained_submission.py](verify_trained_submission.py) |
| Benchmark a complete explicit-path offline workload | [benchmark_offline_ensemble.py](benchmark_offline_ensemble.py) |

Training, data preparation and image-level audits require the matching authorized private inputs. Historical builders and finalizers can write report defaults or allocate quota if their cloud kernels are pushed. Read arguments/configuration and [reproduction instructions](../docs/REPRODUCING.md) before running them. Do not overwrite hash-locked reports to review the assignment. The included result verifier checks aggregates; it is not a new inference evaluation.
