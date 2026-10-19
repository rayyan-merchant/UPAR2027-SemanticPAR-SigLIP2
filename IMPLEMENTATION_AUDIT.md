# Implementation audit — 2026-10-19

Inspected the complete starter README, all 12 Python modules, tests, requirements,
submission template, local official README and Track 1 sample API and metadata,
and actual annotation headers/rows before making implementation changes.
The requested GitHub repository is empty (GitHub connector response); the local
D: starter is the implementation source. Original data and starter stay read-only.

## Blocking correctness findings

1. **Patch flattening is wrong.** Starter uses channel/row/column ordering inside
   patches. HF SigLIP2 uses row/column/channel. This invalidates pretrained vision
   features despite compatible checkpoint shapes. Fix and run real-weight parity.
2. **Checkpoint dimensions are assumed.** Native defaults must be checked against
   downloaded config, including positional-table size, activation and epsilon.
3. **Parity gate is insufficient.** One image and only maximum final error; no
   layer diagnostics, text initialization check, mixed precision or persisted gate.
4. **Resume is not resume.** Saves scoring (EMA) weights without optimizer, scaler,
   raw model, RNG, sampler RNG, EMA or best score. A restart can overwrite best.
5. **Invalid NaN labels poison BCE.** Clamping NaN then multiplying by zero still
   yields NaN. Sanitize targets before BCE and mask finite binary labels explicitly.
6. **Scorer unverified.** Local reference checkout contains no organizer scorer or
   ingestion program. Starter conventions for empty classes/rows, strict >0.5 and
   macro aggregation must not be called exact until differential tests are run.
7. **No enforced data/parity/throughput gates before training.** Add checked evidence
   and refuse full training when gates are missing, stale or unsuccessful.

## Training and inference findings

- Head warmup does not freeze final normalization; partial gradient accumulation
  underweights the final group; EMA updates even after an AMP skipped step.
- Only EMA is validated; raw model comparison, validation loss and attribute
  confusion/precision/recall diagnostics are missing.
- Position cache can become stale after loading/updating weights. Clear cache on
  mode/device/state changes; avoid GPU synchronizations for CPU-known shapes.
- Inference declares BATCH_SIZE but does not chunk input, fails on empty batches,
  publishes global model before load succeeds, and does not reject nonfinite output.
- Benchmark excludes warmup and JSON serialization from total runtime and presents
  an extrapolation as a time-limit verdict. Measure separately and label uncertainty.
- Packaging verifies only two filenames; no checkpoint/calibration compatibility,
  original-metadata comparison or trained-evidence gate.
- Gaussian blur is present, but requested mild JPEG degradation and very light
  erasing are absent. Add conservatively without changing architecture.
- Existing tests: five shape/basic tests only, no backward, parsing, checkpoint,
  packaging, real-image end-to-end, offline subprocess or pretrained tests.

## Environment evidence and unresolved sources

- D:/UPAR2027/.venv is Python 3.10.11 and lacks torch, Pillow, Transformers, pytest.
- Bundled workspace Python is 3.12.14 with NumPy/Pillow, but no torch/Transformers.
- nvidia-smi absent on PATH; Windows GPU inventory query denied. User selected
  Kaggle CLI training; no local CPU training will be started.
- Kaggle CLI 2.2.4 is installed. Normal and elevated account/quota probes fail at
  api.kaggle.com (timeout/DNS/TLS). Account identity, quota and GPU remain unverified.
- Direct git network clone fails DNS. GitHub connector confirms rayyan-merchant.
- CodaBench page retrieval did not return usable rules/scoring content; supplied
  runtime/rules are requirements, not independently verified official evidence.

## Sources inspected

- https://github.com/speckean/UPAR-Challenge-2027
- https://huggingface.co/google/siglip2-base-patch16-naflex
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip2/image_processing_siglip2.py
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip2/modeling_siglip2.py

Fix status and actual test results are tracked in STATUS.md. This audit establishes
findings; it does not certify pretrained parity, official scoring or training.
