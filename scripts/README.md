# Research utilities

| File | Purpose |
|---|---|
| [verify_assignment_results.py](verify_assignment_results.py) | Validate the published result table and comparison decisions without ML dependencies |
| [check_assignment_release.py](check_assignment_release.py) | Check the tracked hand-in files and local documentation links |
| [build_assignment_report.py](build_assignment_report.py) | Rebuild the formatted PDF from its editable source |
| [prepare_lodo_diagnosis.py](prepare_lodo_diagnosis.py) | Prepare isolated source/held-out folds from authorized inputs |
| [generalization_analysis.py](generalization_analysis.py) | Metric and calibration helpers used by evaluation and tests |
| [summarize_lodo.py](summarize_lodo.py) | Aggregate LODO private prediction artifacts |
| [analyze_dinov3_frozen.py](analyze_dinov3_frozen.py) | Independent confusion/masking and DINO evaluation helpers |

The image-level audit scripts require authorized private artifacts and may reference the original research workspace's operational layout. They are not needed to check the published CSV. Internal account, upload, cloud-building and exploratory helper scripts are excluded from this assignment repository.
