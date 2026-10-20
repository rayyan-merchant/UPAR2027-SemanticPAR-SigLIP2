> 2026-10-20 staging recovery supersedes the earlier automatic three-fold plan.
> User verified latest log Staged60000/131076 at35703.7s, then clicked Stop.
> Saved server STATUS independently confirms staging before all training calls.
> API now reports CANCEL_ACKNOWLEDGED; cancellation verified, zero GPU reserved.
> No fold result exists. CPU-only archive probe v2 COMPLETE (900 images/0.601s).
> PA-only archive-backed runner is prepared, not GPU-tested or launched.
> Read STAGING_RECOVERY.md for actual measurements, watchdogs and revised budget.
> Do not restart any GPU training before user review of that schedule.

# Three-domain LODO evaluation

2026-10-20 — **private Kaggle version1 RUNNING**, GPU results pending. Accepted CodaBench
submission1003877 and its production checkpoint/calibration remain unchanged.
No replacement submission is part of this diagnostic.

Actual job: [upar2027-three-domain-lodo-diagnosis](https://www.kaggle.com/code/rayyannmerchant/upar2027-three-domain-lodo-diagnosis).
Submitted runtime source matches commit `a73f6b5`; Kaggle derived the final slug
from the title. There is one active job, not three parallel runs. Quiet scheduled
continuation checks status and retrieves results when available; it does not
authorize another competition submission.

## Fixed protocol

| Order / held out | Source domains | Source train | Selection | Calibration | Target |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 / PA100k | Market1501 + PETA | 18,668 | 11,634 | 11,785 | 9,986 |
| 2 / PETA | PA100k + Market1501 | 89,001 | 13,215 | 13,229 | 6,963 |
| 3 / Market1501 | PA100k + PETA | 87,669 | 8,445 | 8,502 | 16,458 |

PA split CSV bytes were reused exactly after verification. All folds retain the
official train/validation distinction. Source-selection/calibration allocation
uses seed2027 SHA256 of decoded pixels; exact duplicates stay together. Remove
source-validation pixels duplicated in source training. Actual cross-partition
pixel overlaps are zero in each fold. Pixel independence does not establish
person-identity or near-duplicate independence.

Each fold starts from the original pretrained native foundation, SHA256
`55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832`,
converted from general-purpose SigLIP2 revision
`b53b807d3a2d5e2b3911292f2d69e5341cdc064c` before any PAR training. This includes
pretrained text-query initialization and an untrained PAR head. It is not the
all-domain PAR checkpoint. Existing real-weight GPU FP32/FP16 implementation
parity is reused only if native code hash and exact torch/CUDA runtime match;
source-only GPU feasibility reruns for every fold. No pretrained download is
needed: the private original kernel outputs contain this native foundation.

Unchanged architecture/loss/recipe: six epochs maximum; one head-only epoch,
upper-four-block finetuning, exponentially weighted BCE from that fold's source
training labels, per-image domain sampling N_domain^-0.5, AdamW, existing
layer-wise LR decay and warmup/cosine schedule, backbone/head LR1e-5/1e-4,
AMP/gradient accumulation/clipping, EMA.9995,256patches, batch16×4, seed2027.

Checkpoint selection uses only source-selection raw/EMA provisional
dataset-macro Challenge_Avg. Global40 thresholds use only disjoint
source-calibration labels and the original fitting objective/grid/regularizer.
SHA256-lock checkpoint/configuration/splits/calibration before any target model
inference. The target's raw and calibrated predictions come from the same fixed
checkpoint. Interrupted evaluation reuses any atomically saved target
predictions; it never searches target thresholds or alternate checkpoints.

Both legacy harmonic-of-mean-P/R and mean-per-image-F1 conventions, zero-support
mA sensitivity and macro aggregation-order sensitivity are retained. Exact
organizer scorer edge cases remain private, so no exact equivalence is claimed.

## Resource estimate and recovery

Read-only Kaggle quota check:26.01GPUhours remaining. Previous training kernel
is complete. Estimate3.5–5GPUhours for all three folds, including setup and
source/target inference; actual allocation/staging speed is unverified until
the new job starts. Working disk gate requires8GiB free. Three independent
best/last states require about4GB, with roughly1GB transient atomic-save space.

Source-size differences matter: expected six-epoch sampled images/attempted
updates are112,008/1,752;534,006/8,346;526,014/8,220. Total1,172,028images and
18,318attempted updates. Actual successful updates, skipped AMP steps, per-epoch
images and timings are recorded rather than inferred from epochs alone.
Small-source EMA lag/undertraining is a limitation, not an architecture failure.

One private Kaggle job stages official images once, verifying every byte hash
against the prior complete decode audit, then runs PA→PETA→Market sequentially.
Staging uses allowed official data for integrity only; target images are not
decoded/analyzed or sent through a model before source locks. Persistent best
and complete epoch-boundary resume states are separate. Explicit recovery from
an earlier LODO job verifies the fixed split/configuration and never resumes
the all-domain PAR job. No large CPU training is allowed.

## Verified preparation

- Full local suite:27passed in81.29s, including three synthetic domain-isolation
  cases, duplicate-group splitting/reuse, changed-threshold lock rejection and
  convergence-label checks. Existing training math is unchanged; added counters.
- Per-fold CSVs/receipts are local under `outputs/lodo/splits/`.
- Public split hashes/configuration and embedded source hashes:
  `reports/lodo/EXPECTED_SPLITS.json`, `EXECUTION_PLAN.json`,
  `KERNEL_BUILD_RECEIPT.json`.
- CPU source-supervision audit: all40attributes have both positive and negative
  source-training examples in every fold. PA-held-out lower-orange is especially
  sparse: only4positives, versus255/259 in the other two source folds. Thus
  entirely absent training labels do not explain all potential transfer failures;
  rarity and source diversity still matter. Counts are in
  `SOURCE_SUPERVISION_PRECOMMIT.json`; no target model outcome was used.
- AMP/checkpoint API patterns reuse existing tested training code and official
  [PyTorch AMP examples](https://docs.pytorch.org/docs/2.4/notes/amp_examples.html)
  and [general checkpoint guidance](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html).

## Results — pending actual execution

| Held out | Best source score | Target raw | Target calibrated | mA / F1 / P / R / label F1 | Calibration gain | Convergence / compute |
| --- | --- | --- | --- | --- | --- | --- |
| PA100k | pending | pending | pending | pending | pending | pending |
| PETA | pending | pending | pending | pending | pending | pending |
| Market1501 | pending | pending | pending | pending | pending | pending |

Macro/worst-domain scores, source-to-target gaps, repeated attribute failures,
sparse/missing source supervision and the single A–E recommendation must be
filled from completed artifacts. No result or recommendation is invented before
those measurements. The initial same-domain .865 versus official .77045 gap
motivates this protocol; no hidden leaderboard data enter these runs.
