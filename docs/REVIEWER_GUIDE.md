# Reviewer guide

This repository is the assignment's code and results submission. The evaluator needs access to this private GitHub repository; no ZIP download is part of the hand-in requirement. Begin at the root README. No cloud account, dataset or GPU is needed to read the report or verify its aggregate result table.

## Five minute reading path

1. Read the [brief task and approach](../assignment/TASK_AND_APPROACH.md).
2. Open the [three-page report](../assignment/UPAR2027_Assignment_Report.pdf), or its [GitHub-readable text](../assignment/UPAR2027_Assignment_Report.md). It tells the story from initial submission through transfer diagnostics, local model improvements and final research conclusions.
3. Inspect the [official receipt for 1003877](../reports/CODABENCH_SUBMISSION_RECEIPT.json) and the [complete result table](../assignment/RESULTS.csv). Official, local LODO and source-selection records are explicitly distinguished.
4. Browse the [SigLIP2 code guide](../upar/README.md), [DINOv3 code guide](../dino/README.md) and [tests](../tests/README.md).
5. Use the [complete experiment record](../assignment/EXPERIMENTS_AND_DECISIONS.md) and [evidence index](EXPERIMENT_INDEX.md) to trace any decision or number.

## Verify the retained results

After cloning with an authorized GitHub account, run this from the repository root:

```bash
python scripts/verify_assignment_results.py
```

Expected checks include:

```text
passed: true
records: 73
official_entries_confirmed: 1
original_submission_id: 1003877
baseline_hash_locks: unchanged
new_training_started: false
new_official_upload: false
```

This is a standard-library check of saved aggregate evidence, transcription and arithmetic. It is not a new raw-prediction audit or re-training run. For the 71-test CPU suite, use the installation steps in the root README. See [reproduction instructions](REPRODUCING.md) for full private-data-dependent workflows.

## What was actually selected

The original SigLIP2 system is the only confirmed official evaluation, Challenge_Avg 0.770450. Its accepted entry, checkpoints and calibration were preserved. Equal-domain sampling and Candidate C were rejected. Candidate A's frozen ensemble and Candidate B's adapted ensemble improved local transfer but missed their respective declared advancement margins. Separate exploratory authorizations led to final all-domain A/B training and packages; they did not change the negative research gates.

The original SigLIP2 model was submitted successfully. A later upload reset during the reported site outage, yielding no confirmed ID, and further planned official evaluation was not completed. A and B improved every local held-out domain, supporting an expectation of better generalization. B's local LODO macro 0.802508 remains a local result, however, and cannot be presented as its unobserved official score.

## Submission and access checklist

- Submit the repository URL shown in the root README as the code link.
- Grant the evaluator access to the private repository before assessment. Visibility is intentionally unchanged.
- The report and result files are committed on `main`; they do not depend on local absolute paths for viewing.
- Keep restricted datasets, private predictions, model weights and credentials outside Git. Their provenance/hashes and access boundaries are documented.
- Do not launch cloud builders merely to review the assignment. The supplied verification commands start no training or competition upload.
