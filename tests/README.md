# Test guide

From the repository root, install the CPU dependencies described in the [README](../README.md#run-the-code-checks), then run `python -m pytest -q tests`. The recorded local run passed **71 tests**; environment and duration are in [LOCAL_VERIFICATION.json](../assignment/LOCAL_VERIFICATION.json). Tests use small tensors, generated fixtures and local contracts, not a cloud training job.

| Coverage | Tests to inspect |
|---|---|
| Labels, masks, loss and basic model behavior | [test_core.py](test_core.py), [test_correctness.py](test_correctness.py) |
| Visual/preprocessing reference behavior | [test_vision_reference.py](test_vision_reference.py), [test_dino_v3.py](test_dino_v3.py) |
| Sampling and unchanged BCE frequencies | [test_sampling_comparison.py](test_sampling_comparison.py) |
| LODO isolation and source calibration audits | [test_lodo.py](test_lodo.py), [test_dino_audit.py](test_dino_audit.py) |
| Archive staging and resource watchdogs | [test_archive_stage.py](test_archive_stage.py), [test_resource_policy.py](test_resource_policy.py) |
| Frozen/partial regional model and final training invariants | [test_dino.py](test_dino.py), [test_limited_dino.py](test_limited_dino.py), [test_final_a.py](test_final_a.py), [test_final_b.py](test_final_b.py) |
| Isolated degradation augmentation | [test_robustness.py](test_robustness.py) |
| Offline API, raw mixing, thresholds, attribute order and descriptor precision | [test_submission.py](test_submission.py), [test_ensemble_submission.py](test_ensemble_submission.py), [test_finetuned_ensemble_submission.py](test_finetuned_ensemble_submission.py) |
| Generalization metric/calibration analysis | [test_generalization_analysis.py](test_generalization_analysis.py) |

A CPU code-test pass is distinct from reference checks on actual pretrained weights, executed training, measured accuracy, exact A10 timing or official competition acceptance. Those evidence types have separate receipts. The standard-library [assignment results verifier](../scripts/verify_assignment_results.py) also works without installing torch or possessing datasets/weights.
