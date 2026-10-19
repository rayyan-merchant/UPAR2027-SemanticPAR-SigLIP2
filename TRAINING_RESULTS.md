# Actual training results — Kaggle version 4

The single selected model completed 12 epochs. Best validation checkpoint: EMA, epoch 7. No CodaBench submission has occurred.

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
