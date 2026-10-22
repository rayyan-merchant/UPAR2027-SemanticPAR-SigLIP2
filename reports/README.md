# Experimental evidence

This directory retains aggregate results, source curves, support/error tables, environment records, hash locks and execution receipts. It contains no trained weight payloads, image pixels, feature caches or private sample-level prediction arrays. Begin with the [assignment result table](../assignment/RESULTS.csv) and [complete experiment record](../assignment/EXPERIMENTS_AND_DECISIONS.md).

| Evidence | Location |
|---|---|
| Only confirmed official score, original 1003877 | [CODABENCH_SUBMISSION_RECEIPT.json](CODABENCH_SUBMISSION_RECEIPT.json) |
| Original executed training configuration, curves and validation | [kaggle_v4/](kaggle_v4/) |
| Staging recovery benchmarks | [staging_recovery/](staging_recovery/) |
| Original strict LODO metrics and source support | [lodo/results/](lodo/results/) |
| Equal-domain comparison controls/exposure/results | [lodo/equal_domain/](lodo/equal_domain/) |
| Frozen DINOv3, fixed blend and independent checks | [dino_v3/results/](dino_v3/results/) |
| B partial fine-tuning LODO | [competitive_final/candidate_b/](competitive_final/candidate_b/) |
| C augmentation LODO | [competitive_final/candidate_c/](competitive_final/candidate_c/) |
| Final B source evaluation, checkpoint checks, parity and failed runtime probe | [competitive_final/final_b_results/](competitive_final/final_b_results/) |
| Failed A upload without a new submission ID | [competitive_final/CANDIDATE_A_UPLOAD_RECEIPT.json](competitive_final/CANDIDATE_A_UPLOAD_RECEIPT.json) |

Evidence is stage-specific. A historical plan or pending receipt is not a completed result; final state is recorded in [STATUS.md](../STATUS.md) and the [reviewer guide](../docs/REVIEWER_GUIDE.md). Read scorer convention and unsupported-label caveats before comparing metrics. Baseline locked files are preserved byte-for-byte; `python scripts/verify_assignment_results.py` checks them from the repository root.
