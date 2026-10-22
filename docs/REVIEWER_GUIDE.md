# Reviewer guide

1. Read the [task description](../assignment/TASK_AND_APPROACH.md) and [three-page report](../assignment/UPAR2027_Assignment_Report.pdf).
2. Browse the [compact results CSV](../assignment/RESULTS.csv) and [scientific decision record](../assignment/EXPERIMENTS_AND_DECISIONS.md).
3. Inspect [SigLIP2](../upar/), [DINOv3](../dino/) and the [tests](../tests/README.md).
4. Run `python scripts/verify_assignment_results.py` from the repository root for aggregate-table checks, or follow the [reproduction guide](REPRODUCING.md) for CPU tests and authorized-input training.

Only original submission 1003877 has an official score, 0.770450. Later A/B ensembles improved local transfer; follow-up official evaluation was not completed during the reported site outage. Local LODO and source-selection scores are marked separately. One seed, unequal source sizes and repeated development-target inspection limit generalization claims.

The results verifier checks the published aggregate table's consistency, not raw private predictions. Internal execution receipts, plans, logs and analysis dumps are retained privately and are not part of this code hand-in. Dataset images and trained weights require separate authorization and are not redistributed here. The repository remains private; evaluator access is granted by the submitter.
