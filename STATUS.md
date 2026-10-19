# Status — 2026-10-19

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
- User-approved quiet follow-up is ACTIVE every 15 minutes (automation
  `finish-upar-kaggle-submission`). It checks outputs and continues evaluation,
  calibration and packaging; notifications are limited to meaningful results,
  failures or required user action.
- **Actual training completed 12 epochs. Best checkpoint: epoch 7 EMA.**
  Validation Challenge_Avg 0.856185; duplicate-excluded 0.856180. Global calibration
  reached 0.864952 on its own fit rows (in-sample, provisional scorer).
- Trained candidate ZIP and resume checkpoints exist on Kaggle; local downloads
  and final trained offline/package verification are underway. No CodaBench
  submission or verified A10 300-second result exists. See TRAINING_RESULTS.md.

## Blockers and limitations

- CodaBench `make_programs_available=false`. Organizer program metadata/download
  is not exposed (403 on program API). Published metric formulas are verified,
  but exact invalid/empty-class edge behavior cannot yet be differentially tested.
  Local scores remain provisional. Do not mark exact scorer equivalence passed.
- Kaggle GPU parity, feasibility and all 12 training epochs passed. Local
  target-version trained-package verification and final timing checks remain.
- Local tests used Python 3.12; Kaggle used Python 3.13/torch 2.11. A Python
  3.11 target-version inference environment is being prepared.
- Exact duplicate checks do not exclude near-duplicate/person-identity leakage.
- Kaggle measurements cannot certify the A10 300-second evaluation wall limit.

## Next actions

Finish retrieving the trained candidate and resumable checkpoints. Verify the
trained package offline under target torch 2.4.1, separate benchmark setup cost
from the supplied-sample API workload, and verify the ZIP before CodaBench upload.

`reports/data_mapped/DATA_AUDIT.md` is the canonical data report; `reports/data/`
preserves the initial failed path audit. `reports/tests.xml` records local checks.
