# Actual training results — Kaggle version 4

The single selected model completed 12 epochs. Best validation checkpoint: EMA, epoch 7. CodaBench submission1003877 is Finished, official Challenge_Avg .770450. The same-dataset validation score did not predict unseen-domain performance; see GENERALIZATION_DIAGNOSIS.md. No further training or resubmission has been started.

## Official score and diagnosis (2026-10-20)

Official scoring archive: mA .773647, instance F1 .767279, precision .741799,
recall .807915, label F1 .583226. All29,248 images processed in109.91s,
scorer exitCode0. Server ZIP checksum matches the accepted candidate exactly.
Raw-to-official gap8.573points; calibrated-to-official gap9.450points.
Disjoint calibration folds on allowed validation predictions achieve .863482;
fit optimism only.147point. This does not isolate model-selection optimism or
unseen-domain shift. Local scoring convention is also inconsistent with the
official per-domain P/R/F1 relationship; exact private scorer remains unavailable.
The detailed CPU audit and prepared, unexecuted true-LODO design are documented
in GENERALIZATION_DIAGNOSIS.md. Original checkpoint/calibration remain preserved.

Historical timing/pending statements below describe earlier stages and are
superseded by the completed official evaluation and diagnosis above.

## Verified gates

- Complete official-image audit passed; known duplicate review passed.
- 20 tests passed on Kaggle.
- Real pretrained GPU FP32 and FP16 parity passed on four images with four aspect ratios: all recorded maximum/mean absolute and relative L2 errors were zero, including every Transformer block.
- Actual forward/backward feasibility passed at microbatch 16, accumulation 4, effective batch 64.
- Pipeline wall time was 12,661.65 seconds (about 3.52 hours), including audits, training, validation, calibration and packaging.

## Validation (provisional scorer)

| Setting | Dataset-macro Challenge_Avg | mA | Instance F1 |
|---|---:|---:|---:|
| Best checkpoint, threshold 0.5 | 0.856185 | 0.834468 | 0.879118 |
| Duplicate-excluded, threshold 0.5 | 0.856180 | 0.834463 | 0.879114 |
| Calibrated, duplicate-excluded (in-sample) | 0.864952 | 0.854915 | 0.875367 |

| Domain | Before calibration | After calibration (in-sample) |
|---|---:|---:|
| Market1501 | 0.848187 | 0.853922 |
| PA100k | 0.845154 | 0.858075 |
| PETA | 0.875198 | 0.882858 |

Calibration excluded two known duplicate validation rows. The same validation split selected the checkpoint and fitted thresholds, so calibrated scores are in-sample. Exact organizer scorer edge cases remain unavailable for differential verification. These are known-domain validation results, not hidden-domain leaderboard scores.

## Candidate and timing

The trained candidate ZIP was downloaded and its ZIP CRC, exact original metadata, trained epoch7 EMA checkpoint, semantic query shape and offline API passed on Python3.11.16/torch2.4.1+cpu and Python3.12/torch2.4.1. Network and optional HF/timm/requests imports were blocked. Two real images and a 35-image chunked batch returned finite ordered probabilities. Single/batch maximum absolute difference was 3.28e-7. Archive SHA256: `76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308`; best checkpoint SHA256: `18a48956eac97d16709e355d57e747a9944c872f844e20645d1dc5aae86ae0d6`. Independent recomputation reproduced both validation scores and the calibration prediction hash. Final GPU timing verification is pending; no CodaBench submission is claimed.

Measured trained inference on Tesla T4: 2,000 images in 13.4243 seconds after warmup (148.984 images/s), with decoding, preprocessing and serialization in the timed loop. Model load 1.4863 seconds; warmup 1.0693 seconds. Full process measured 211.6601 seconds. The remaining startup overhead has not been separately attributed; the benchmark enumerates the dataset before inference. Its hypothetical 33,407-image projection was 422.4679 seconds, above 300. Do not certify the official A10 time limit from this run. Separate discovery/setup from supplied-sample inference in the next verification benchmark.

Raw evidence is retained under reports/kaggle_v4/.

## Trained GPU verification follow-up

Inference-only version1 passed blocked-network/optional-dependency CUDA API checks
on six real images across the official domains, 35-sample batching and original
metadata; single/batch maximum probability difference0.001139 (tolerance0.002).
An explicit-path batch32 pass took253.237s for2000images, followed by batch64
13.864s on the same mounted images. This order/cache confound prevents attributing
the difference to batch size. Version2 measures staging from the private input
mount separately, then compares identical derived local image copies. No model
training or architecture change is involved. Its outcomes are pending.
User subsequently signed in; account rayyanmerchant and available competition upload interface were verified. Final timing checks remain before upload.

## Final local-storage measurements

Inference-only version2 completed successfully. Staging2100 official images
(19,513,905 bytes) from the private Kaggle mount into local working storage took
194.272s, reported separately. On the same staged sample list, batch32 processed
2000images in10.2633s (194.868 images/s), complete benchmark process14.5758s;
batch64 took10.9090s (183.335 images/s), complete process15.4839s. Original batch32
was retained, with the identical previously verified archive hash.
The hypothetical33407-image projection at batch32 is175.746s; this is an estimate
on T4/torch2.11, not certification of the actual A10/torch2.4.1 workload or300s
limit. No hidden test images, counts or metadata were used for this estimate.
The subsequent CodaBench evaluation finished at.770450; see the current receipt.

## Official submission receipt

CodaBench accepted the identical verified archive as submission **1003877**,
account rayyanmerchant, page-displayed date2026-10-20 03:36. Current status:
**Finished**, official Challenge_Avg **.770450**, reported inference109.91s.
Server file SHA256 exactly matches the retained ZIP. Preserve this accepted
submission; no automatic replacement. See reports/CODABENCH_SUBMISSION_RECEIPT.json.
