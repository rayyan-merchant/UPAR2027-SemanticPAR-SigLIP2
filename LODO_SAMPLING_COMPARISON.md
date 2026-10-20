# Equal-domain sampling: single predeclared three-fold comparison

Status: private Kaggle version1 RUNNING; no comparison result yet.
Kernel rayyannmerchant/upar2027-lodo-equal-domain-comparison, sourcebe2a8f0.
Provider timeout10800s. Receipt reports/lodo/equal_domain/KERNEL_SUBMISSION_RECEIPT.json.
Do not duplicate an active job. Locked baseline: commit4b60ffc,
LODO_EVALUATION.md and reports/lodo/results. Baseline files are SHA256-pinned in
reports/lodo/equal_domain/BASELINE_LOCK.json and must not be overwritten.

## Intervention and controls

Only methodological change: per-image sampling weight1/N_domain (alpha0),
therefore P(source domain)=.5 for each of two domains. Baseline per-image weight
N_domain^(-.5), therefore domain probability proportional to sqrt(N_domain).
Same sampler replacement and number of draws, seed2027, foundation SHA
55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832,
pretrained semantic queries, architecture, augmentation, source-only raw-row
frequency/BCE weights, upper4 fine-tuning, warmup, optimizer/LRs/cosine, EMA,
AMP,256patches,microbatch16/accum4,workers2,threads2 and six epochs.

All12 CSV partition hashes remain exactly precommitted. Target excluded from
training, frequencies, sampling, selection and calibration. Source-only raw/EMA
checkpoint selection and disjoint source-calibration global40 thresholds remain
unchanged; config/checkpoint/calibration locked before one fixed target inference.
Every fold starts fresh from the original untrained foundation; neither baseline
LODO weights nor the all-domain PAR model are attached as recovery sources.

Exposure instrumentation records counts and unique images per domain per epoch
and cumulatively, without consuming RNG or changing sampled indices. Private
unique index sets are checkpointed with optimizer/EMA/allRNG. Completed-epoch
counts must equal consumed batches; source-only exposure is checked before target
inference. Baseline exposure was not logged: its expected probabilities must be
clearly distinguished from measured new counts. Class-frequency computation and
BCE weights are compared exactly against the baseline source supervision.

## Fixed comparison and adoption rule

| Held out | Locked baseline raw | Locked baseline calibrated |
| --- | ---: | ---: |
| PA100K | .786128375 | .797104355 |
| PETA | .769342036 | .777228370 |
| Market1501 | .813343356 | .817447759 |
| Macro | .789604589 | .797260162 |
| Worst | .769342036 | .777228370 |

Record raw/calibrated Challenge_Avg,mA,instance-F1,P/R,label-F1, attribute groups,
source scores/curves, draw/unique counts and epoch/wall costs. Report PETA and
Market separately. Keep both legacy harmonic-of-mean-P/R and mean-image-F1 /
zero-support sensitivities; exact organizer edge cases remain private.

Before seeing new results, operationalize 'convincing' conservatively as:
calibrated macro gain >=.003, worst-domain gain >=.003, no individual calibrated
domain regression >.005, and positive macro/worst gains under mean-image-F1 too.
These are decision margins, not statistical significance thresholds. If mixed,
negligible, a scorer-dependent reversal, or a gate failure occurs, retain the
original sampler for the final recipe. Inspect mA/F1/P/R contributors and source
convergence; do not move these criteria after seeing outcomes. Single seed cannot
measure training-seed variance. Previously inspected targets make this a single
development comparison, not an unbiased untouched generalization test or a basis
for repeated method search. No hidden competition feedback is used.

## Resource and recovery protocol

Separate private kernel upar2027-lodo-equal-domain-comparison. Reuse existing four
private opaque archives, stage/extract once: prior full CPU34.42s/GPU32.96s.
No extra CPU cloud benchmark or per-image mounted copy/hash/decode. Stage120s /
idle30s; whole pretraining setup600s; train/selection/calibration/target loader
I/O timeout30s. Same measured dual-T4 source-only solo/concurrent gate and
GPU0PA/GPU1PETA, then MarketGPU0 after PA. Separate processes/states/logs.

Target1.5-2.5h; hard3h including setup, any technical retries and recovery.
Provider timeout10800s plus internal deadline/early projection stop. Reserve
>=8GPUh for later work; live prelaunch quota12.50h means >=9.50h after a full
3h comparison. No repeated dependency installs inside folds. Original atomic
epoch-boundary best/last persist privately; abort on infrastructure delay or
resource projection conflict, preserve evidence and do not silently drop folds.

## Academic and competition completion plan

Use the user's deadlines: academic October21, competition October22,2026.
Target report immediately after terminal comparison and independent checks
(allow30-60min CPU collection/analysis). Report the sampler decision and live
quota before any next compute. No final all-domain training or CodaBench upload
is authorized by this diagnostic.

If subsequently authorized and supported by results, allocate2.3-3.0h for one
12-epoch all-domain training run and .5-1.0h for calibration/package/verification;
allow3.5-4.5h total contingency. This is an estimate, based on measured training
throughput and the original3.52h pipeline, not a guaranteed finish time. If this
jeopardizes academic submission, use the accepted model and documented results.

By October21: finish the academic manuscript/report with accepted model facts,
official score, locked baseline and this comparison, methods, dataset/license
provenance and limitations. Obtain actual author/affiliation/team information
and the real submission portal/format; do not invent human details or claim
academic acceptance. Prepare code/config/checksum/factsheet artifacts alongside.
Academic portal submission remains a human/account action unless authorized.

By October22: preserve1003877 as fallback; submit any approved improved candidate
early enough for upload and terminal evaluation, allowing at least2h contingency
before the actual portal cutoff. Verify the exact cutoff/timezone in the portal
before scheduling against it; a calendar date alone is not a precise hour.
Never search the hidden leaderboard, automatically resubmit, email organizers,
or replace the accepted package solely from this experiment.
