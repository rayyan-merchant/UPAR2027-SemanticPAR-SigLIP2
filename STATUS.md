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
- 19 automated tests passed locally under torch 2.4.1+cpu, Python 3.12.14,
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
New pretrained-layer parity and GPU feasibility tools still need actual weights
and GPU execution; passing small-model tests is not pretrained parity evidence.

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
  permitting the single main training run. CLI still reports QUEUED; GPU allocation
  and gate outcomes are pending. Source/metadata receipt: reports/kaggle_v2_receipt.json.
- Real foundation checkpoint download/parity is in progress locally. A bounded
  direct HTTPS download is transferring after hub/Xet stalled. No
  pretrained numerical parity pass has been claimed.
- No verified evidence of main training starting yet. No trained checkpoint, model score, final
  calibration, A10 timing or competition submission exists yet.

## Blockers and limitations

- CodaBench `make_programs_available=false`. Organizer program metadata/download
  is not exposed (403 on program API). Published metric formulas are verified,
  but exact invalid/empty-class edge behavior cannot yet be differentially tested.
  Local scores remain provisional. Do not mark exact scorer equivalence passed.
- Local environment is CPU-only torch. GPU parity and benchmark must pass on
  Kaggle before full training. Kaggle allocation is not guaranteed by quota.
- Tests used Python 3.12, not the official Python 3.11 container.
- Exact duplicate checks do not exclude near-duplicate/person-identity leakage.
- Kaggle measurements cannot certify the A10 300-second evaluation wall limit.

## Next actions

Finish private upload, launch the CLI-only GPU pipeline, inspect actual parity and
throughput evidence, train the single selected architecture only after gates pass,
evaluate raw/EMA weights, calibrate using duplicate-excluded validation, then
benchmark and inspect candidate ZIP before any CodaBench upload.

`reports/data_mapped/DATA_AUDIT.md` is the canonical data report; `reports/data/`
preserves the initial failed path audit. `reports/tests.xml` records local checks.
