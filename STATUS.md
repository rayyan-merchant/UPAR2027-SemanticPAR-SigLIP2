> 2026-10-20 staging recovery supersedes the earlier automatic three-fold plan.
> User verified latest log Staged60000/131076 at35703.7s, then clicked Stop.
> Saved server STATUS independently confirms staging before all training calls.
> API now reports CANCEL_ACKNOWLEDGED; cancellation verified, zero GPU reserved.
> No fold result exists. CPU-only archive probe v2 COMPLETE (900 images/0.601s).
> PA-only archive-backed runner is prepared, not GPU-tested or launched.
> Read STAGING_RECOVERY.md for actual measurements, watchdogs and revised budget.
> Do not restart any GPU training before user review of that schedule.

# Status — 2026-10-20

## Current execution: three-domain LODO; preserve submission

- User authorized all3six-epoch diagnostic folds, sequential PA100k→PETA→Market.
  **Private Kaggle version1 RUNNING**:
  rayyannmerchant/upar2027-three-domain-lodo-diagnosis. Actual source commit
  a73f6b5; returned slug differs from initially requested shorter slug.
  Do not duplicate an active run or push a new version while it is running.
- Prepared all3source train/selection/calibration/target splits with zero
  cross-partition decoded-pixel overlap; PA CSVbytes reused exactly.
  Original untrained foundation SHA55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832
  only; never use original PARbest/last as initialization or recovery.
- Quota26.01GPUhours before launch; estimate3.5–5GPUhours total. ActualGPU
  staging/training/metrics remain pending evidence.27tests passed locally
  before launch. LODO_EVALUATION.md and reports/lodo contain protocol/receipts.
- Automatic follow-up is ACTIVE for this diagnosis only: retrieve actual
  outputs, verify all3folds, finish report and one evidence-based recommendation.
  No competition upload, architecture/method change or hidden-score tuning.
- Preserve all prior canonical artifacts and original D: datasets. Best/last
  LODO states persist privately in Kaggle outputs; download small diagnostics
  first to respect C: capacity. No three-fold result is claimed yet.

- CodaBench submission **1003877 Finished**: official Challenge_Avg **0.770450**,
  mA .773647, instance F1 .767279, precision .741799, recall .807915.
  Scoring report: all29,248 images processed through predict_batch in109.91s,
  scorer exitCode0. Public aggregate results only were inspected.
- Server-downloaded Submission File SHA256 exactly equals the retained final ZIP:
  `76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308`.
  Independent best/model/calibration/log/prediction/archive copies and hashes are
  retained in outputs/preserved_1003877. Accepted artifacts remain unchanged.
- Earlier diagnosis involved no training/model/threshold/upload changes. New
  source-held-out training above is now explicitly authorized. Production
  checkpoint/calibration and accepted submission remain unchanged.
- Completed CPU diagnosis in GENERALIZATION_DIAGNOSIS.md and aggregate evidence
  reports/generalization_1003877/. Disjoint five-fold calibration score .863482
  versus .864952 fitted: .147point optimism; known-domain OOF gain .730point.
  This is calibration-only cross-validation; the fixed checkpoint used all val.
- Calibration raises known-domain FP32,706→39,303 and lowers FN28,096→25,272.
  Local F1 convention differs from the official per-domain aggregate relationship;
  measured alternative-definition sensitivity .19point, joint zero-support
  scenario1.04points. Private exact scorer remains unverified.
- Actual48-image validation-versus-ZIP audit passed: preprocessing/model FP32/
  threshold offsets/padding all zero measured error; all33,407 annotation rows
  align with stored predictions. CPU-vs-stored GPU AMP mean probability error
  .000350,max .015984; one calibrated decision differs among1,920 cells.
- Initial fresh-foundation PA-held-out six-epoch split, now reused by LODO job:
  train18,668; source selection11,634; separate calibration11,785; target9,986.
  No cross-partition decoded-pixel overlap. Current PAR weights cannot initialize
  this run because they saw PA. Estimate .5–1GPUhour,1.25h running cap, uncertain
  queue/setup. Quota check26.01h remaining; no allocation claimed.
- Next: report findings first, then use one source-held-out diagnostic to test
  the protocol hypothesis before any model/loss/threshold/interpolation change.
  C: storage is tight; only the explicitly approved verified358MB duplicate
  candidate-chunk folder was removed. Preserve canonical ZIP/checkpoints/data.
- Full local suite now22passed in53.52s. Restricted pytest discovery to tests/
  after an initial unrestricted run collected a stale extracted offline script.

The remaining historical notes below record earlier stages; their pending-upload
and active-monitor wording is superseded by the current decision above.

## Completed and verified

- Full starter code/README/requirements/tests and official Track 1 sample inspected.
- GitHub connector identity: `rayyan-merchant`; requested repository initially empty.
- Kaggle CLI 2.2.4 authenticated as `rayyannmerchant`; GPU quota 30.00h remaining
  at the live pre-training check, weekly refresh shown as 2026-10-24T00:00:00.
- Original data and local starter on D: remain unchanged.
- Both official CSVs match the exact ordered 40-attribute vocabulary: 97,669 train,
  33,407 validation, no invalid-label cells in this release.
- Full extracted file inventory: PA100k 100,000; Market1501 61,295; PETA 19,000.
  The earlier 16,524 PETA count excluded 2,476 BMP files.
- Full read-only audit decoded and hashed all 131,076 annotated images.
- Resolved missing 15,631 PETA canonical paths through the official mapping.
- No annotation path overlap; no cross-split byte duplicates. One decoded-pixel
  group crosses splits: MIT PETA 08572 (train), 08573 and 08574 (validation),
  with identical labels. Official splits retained; calibration excludes two rows.
- 20 automated tests passed locally under torch 2.4.1+cpu, Python 3.12.14,
  Transformers 4.57.1. Includes native/installed-HF processor parity, backward and
  padding masks, NaN masking, sampler probabilities, known metric examples,
  random-weight native/HF vision equivalence, one-step Adam resume equivalence,
  blocked-network offline inference on a real
  pedestrian image and temporary synthetic-checkpoint ZIP structure.
- Official CodaBench public API rules retrieved: 300s, A10 24GB, strict >0.5,
  dataset-equal averaging, original metadata, offline torch 2.4.1/Python 3.11.

## Corrected code (verification boundaries)

Patch RGB ordering, invalid-label sanitization, backbone-config assertions,
position-cache invalidation, head-only freeze, partial accumulation scaling,
AMP-skipped-step EMA, raw/EMA checkpoint selection, full epoch-boundary resume,
batch chunking/empty input/nonfinite rejection, calibration ordering/provenance,
benchmark serialization/cold-start reporting, data/parity/GPU training gates.
Real pretrained CPU and GPU parity and GPU feasibility passed on corrected source.
Actual evidence is recorded below and in TRAINING_RESULTS.md.

## Current execution

- Private Kaggle official-data archive prepared: 702,089,736 bytes, 131,076 images.
  SHA256: `4d4eb2c66c16fc132fe3249e88847f526dbb9f7882071397cd1033a52a606097`.
- Kaggle CLI confirmed successful private upload. Dataset
  `rayyannmerchant/upar2027-official-private` now reports ready; its image files
  are accessible through CLI. Kaggle expands the ZIP into `official_data/`.
- Private kernel version 1 failed its missing-data gate while the data source
  was still processing. Actual hardware: two Tesla T4 GPUs, 15.6GB each,
  torch 2.11.0+cu128/CUDA 12.8. No training occurred. Quota after failure: 29.99h.
- Private kernel version 2 was successfully pushed with the ready data source.
  It reruns data/tests/pretrained parity/inference/training feasibility before
  permitting main training. Its diagnostic source was superseded by version 3
  after the local real-weight check exposed and resolved the hook issue.
- **Private kernel version 3 ended ERROR at the AMP parity gate.** Source
  matched commit `9c5a0e0`. Its full data audit and 19 tests passed. GPU FP32
  preprocessing and all vision stages matched exactly on four real images. AMP
  errors exceeded unchanged tolerances (max 0.662–5.659; relative L2 0.0126–0.0430).
  Training and feasibility never started. Evidence: reports/kaggle_v3/.
- Corrected native AMP position casting: preserve FP32 positional embeddings and
  residual sums like Hugging Face. New CPU BF16 regression failed before the fix
  and passed afterward; full local suite now 20 passed. Added AMP layer diagnostics.
  **Private kernel version 4: COMPLETE** per CLI. All 26 submitted source
  files match commit `bc987db`, including original metadata and attached private
  data source (reports/kaggle_v4_receipt.json). GPU FP32/FP16 parity passed with zero
  errors on four real images; 20 Kaggle tests and training feasibility passed.
  Quota immediately before version 4: 29.82 GPU hours remaining.
- Real foundation checkpoint (1,500,985,224 bytes) downloaded and SHA256 verified.
  Resolved revision: `b53b807d3a2d5e2b3911292f2d69e5341cdc064c`.
- **CPU FP32 pretrained parity PASSED on four real pedestrian images, four aspect
  ratios.** Preprocessing, patch embeddings, all 12 vision blocks, final outputs
  and text initialization: observed maximum/mean absolute and relative L2 errors
  all zero. Query shape 40x768. Native prepared checkpoint SHA256:
  `dc60844b23b36dc74d7321fa4095d6ecd9b63a072c0282a8c70403bfe1ade2b8`.
  Reports: reports/PRETRAINED_PARITY.json and .md. GPU AMP parity passed on corrected code (reports/kaggle_v4/reports/PRETRAINED_PARITY.json).
- Initial real-weight diagnostic failed because HF's internal transformer does
  not expose hidden_states through the wrapper's capture decorator. Explicit
  reference hooks corrected diagnostics; subsequent real-weight check passed.
- Follow-up `finish-upar-kaggle-submission` is repurposed to three-domain LODO
  diagnosis only; automatic competition resubmission remains prohibited.
- **Actual training completed 12 epochs. Best checkpoint: epoch 7 EMA.**
  Validation Challenge_Avg 0.856185; duplicate-excluded 0.856180. Global calibration
  reached 0.864952 on its own fit rows (in-sample, provisional scorer).
- Private inference-only kernel `rayyannmerchant/upar2027-trained-inference-verification`
  version 1 COMPLETE: trained offline CUDA API passed on six real images and
  35-sample batches, single/batch probability max difference 0.001139.
  Explicit-path batch32 took253.24s/2000 images; subsequent batch64 took13.86s.
  The comparison is confounded by cold/warm mounted image access. Version2
  COMPLETE: storage staging measured separately; batch32 local-file inference
  2000 images/10.263s (194.868 images/s), batch64 10.909s. Original batch32
  retained. Staging2100images/19.51MB took194.272s, supporting the storage
  bottleneck explanation. No training was invoked.
- Candidate ZIP downloaded (358,029,220 bytes), SHA256
  `76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308`.
  Best epoch7 EMA weights, validation predictions and complete last.pt are
  preserved locally. Resume backup loaded successfully: schema2, epoch12,
  18,324 steps, 85 optimizer parameter states, EMA and all RNG/sampler states.
  Last checkpoint SHA256:
  `64d23096ce17db94f667055c65dfddbf3861844c4e8867e64186ed2eb9259f74`.
- **Actual trained offline verification PASSED on Python3.11.16/torch2.4.1+cpu**,
  and separately Python3.12/torch2.4.1. Two real images, 35-sample chunked batch,
  finite 40 probabilities, correct ordering, exact original metadata/ZIP CRC,
  blocked sockets and optional HF/timm/requests imports. Single/batch max error
  3.28e-7. Report: reports/kaggle_v4/TRAINED_OFFLINE_PY311_TORCH241.json.
- Independently recomputed saved-prediction and calibrated scores and verified
  calibration input SHA256 and the exact two excluded duplicate paths.
- User signed in to CodaBench; authenticated account `rayyanmerchant` and
  competition upload interface verified, 0/3 daily and 0/50 total submissions
  used before upload. **CodaBench submission1003877 is Finished**, score.770450.
  Page date2026-10-20 03:36; last observed daily1/3 and total1/50 used.
  Official scoring report records109.91s inference; see TRAINING_RESULTS.md.

## Blockers and limitations

- CodaBench `make_programs_available=false`. Organizer program metadata/download
  is not exposed (403 on program API). Published metric formulas are verified,
  but exact invalid/empty-class edge behavior cannot yet be differentially tested.
  Local scores remain provisional. Do not mark exact scorer equivalence passed.
- Kaggle GPU parity, feasibility, training and target Python3.11/torch2.4.1
  CPU and GPU trained-package verification passed; final T4 timing is recorded.
- Target Python3.11/torch2.4.1 CPU inference passed. Kaggle GPU runtime uses
  Python3.13/torch2.11; the actual official run finished and reported109.91s
  inference, without a separate hardware/container attestation in the archive.
- Exact duplicate checks do not exclude near-duplicate/person-identity leakage.
- Kaggle measurements alone cannot certify A10 latency; the accepted official
  run now supplies actual workload timing evidence.

## Next actions

Use the diagnosis and prepared PA-held-out plan before new GPU expenditure.
No automatic upload, full all-domain retraining or hidden-score tuning.
Submission ZIP: outputs/submission/UPAR2027_SemanticPAR_SigLIP2.zip (same verified SHA256).

`reports/data_mapped/DATA_AUDIT.md` is the canonical data report; `reports/data/`
preserves the initial failed path audit. `reports/tests.xml` records local checks.
