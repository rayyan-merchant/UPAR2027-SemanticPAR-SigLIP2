# Experiment evidence index

Start with [the three-page assignment report](../assignment/UPAR2027_Assignment_Report.pdf), [the complete experiment and decision record](../assignment/EXPERIMENTS_AND_DECISIONS.md) and [the machine-readable results](../assignment/RESULTS.csv). All official-score claims are limited to original submission 1003877.

| Question | Retained evidence |
|---|---|
| Original architecture and recipe | [Implementation audit](../IMPLEMENTATION_AUDIT.md), [executed configuration](../reports/kaggle_v4/training/training_config.json) |
| Original official evaluation | [Training results](../TRAINING_RESULTS.md), [official receipt](../reports/CODABENCH_SUBMISSION_RECEIPT.json) |
| Why validation failed to predict official score | [Generalization diagnosis](../GENERALIZATION_DIAGNOSIS.md) |
| Slow mounted storage and archive repair | [Staging recovery](../STAGING_RECOVERY.md), [staging reports](../reports/staging_recovery) |
| Strict fresh-foundation LODO baseline | [LODO evaluation](../LODO_EVALUATION.md), [raw aggregate evidence](../reports/lodo/results) |
| Equal sampling rejection | [Controlled comparison](../LODO_SAMPLING_COMPARISON.md) |
| DINO feature extraction and model correctness | [Architecture](../DINO_ARCHITECTURE.md), [frozen results](../DINO_LODO_RESULTS.md) |
| Frozen A complementary ensemble | [Comparison](../ENSEMBLE_COMPARISON.md), [A acceptance](../CANDIDATE_A_ACCEPTANCE.md) |
| B limited fine-tuning and failed advancement margin | [B LODO](../DINO_FINETUNE_LODO.md) |
| C negative augmentation ablation | [C ablation](../SIGLIP_ROBUSTNESS_ABLATION.md) |
| Final exploratory B training and recoveries | [Final training](../FINAL_B_TRAINING_REPORT.md), [recovery](../FINAL_B_RECOVERY.md) |
| Final implementation and model correctness | [B acceptance](../CANDIDATE_B_ACCEPTANCE.md), [implementation evidence](../reports/competitive_final/final_b_results) |
| What actually reached CodaBench | [Submission log](../OFFICIAL_SUBMISSION_LOG.md), [A unsuccessful upload receipt](../reports/competitive_final/CANDIDATE_A_UPLOAD_RECEIPT.json) |

Older reports are stage-specific records and can contain historical pending statements. Use the assignment decision record and current STATUS for final state. Do not rerun old builder scripts or overwrite locked baseline reports while reading them.
