# Kaggle CLI execution

Account verified live: `rayyannmerchant`. All data and kernels must remain private.
Use absolute paths with the installed CLI; a relative dataset upload path failed
inside the CLI's upload-state handling. Never add `--public` or `-u`.

## Data staging (originals stay read-only)

The full audit is in `reports/data_mapped/`. PETA canonical annotation paths are
resolved through the official mapping; the archive copies original bytes under
those paths, matching the official script's renaming behavior. No images were
redownloaded or re-encoded. Only annotated training/validation images are staged.

```powershell
python scripts/prepare_kaggle_data.py --audit-dir reports/data_mapped --data-root D:/UPAR2027/official/data --owner rayyannmerchant --output C:/absolute/staging/data
kaggle datasets create -p C:/absolute/staging/data -t
kaggle datasets status rayyannmerchant/upar2027-official-private
```

This is private academic competition processing, subject to original image terms.
PETA is research-only, annotations CC BY-NC-SA 3.0, PA100K CC BY 4.0; original
Market1501 terms remain applicable. The Kaggle license field `other` does not
grant downstream redistribution rights. Do not publish or share the data source.

## One architecture, gated pipeline

```powershell
kaggle quota
python scripts/build_kaggle_kernel.py --owner rayyannmerchant --output C:/absolute/staging/kernel
kaggle kernels push -p C:/absolute/staging/kernel --accelerator NvidiaTeslaT4 -t 43200
kaggle kernels status rayyannmerchant/upar2027-siglip2-semanticpar
kaggle kernels logs rayyannmerchant/upar2027-siglip2-semanticpar
```

The CLI sends an embedded source snapshot, with a private dataset attached.
`NvidiaTeslaT4` requests Kaggle's T4 x2 configuration; the current training code
uses GPU 0 only. GPU assignment is established by the actual hardware report.
Quota is a ceiling, not a guarantee of allocation. The script does not run CPU
training when GPU allocation fails.

The pipeline re-audits extracted images, runs tests, verifies all native vision
layers and text-query initialization with real pretrained weights, verifies GPU
AMP parity, benchmarks six forward/backward batches, and starts main training only
after those gates pass. If batch 16 is out of memory, batch 8 with accumulation 8
preserves effective batch 64. If estimated training alone exceeds eight hours, it
stops for review instead of committing the whole GPU budget.

The scorer follows the documented formulas; exact organizer-program differential
verification remains unavailable because CodaBench disables program downloads.
Local checkpoint-selection scores must be described as provisional.

## Persistence and resume

```powershell
kaggle kernels output rayyannmerchant/upar2027-siglip2-semanticpar -p C:/absolute/saved-run
```

Save completed run outputs before changing the kernel. `runs/siglip2/last.pt`
contains raw model, optimizer, AMP scaler, EMA, RNG/sampler state, completed epoch,
global schedule step and historical best score. `best.pt` is separate inference
weights. Resume is at the last completed epoch; arbitrary mid-epoch replay is not
claimed. Multiworker augmentation trajectories are reproducible with unchanged
configuration and epoch boundaries, subject to CUDA nondeterminism.

Attach the previous output as a kernel source using `--resume-source owner/slug`
when building a new kernel. The runner finds one `last.pt`, preserves existing
best artifacts and resumes without resetting the best score. Choose a distinct
source/output slug if the service disallows a self-reference. Training budget,
optimizer and sampling configuration must match; the loader rejects mismatch.
Do not launch concurrent versions against the same output source.

Kaggle saves notebook-version outputs on completion. A hard platform failure can
still lose unpersisted progress. Epoch-end checkpoints and an early nine-hour stop
reduce this risk; they do not guarantee durability during arbitrary termination.

## Candidate submission

The pipeline calibrates after training, excluding the two known duplicated
validation rows. Reported calibration fit scores are in-sample. It creates
`submission_candidate.zip` only after actual training, then runs decoding,
preprocessing, inference and JSON-serialization benchmarks. A T4 projection for
a hypothetical image count does not certify the official A10 runtime.

Before CodaBench upload, review actual logs, checkpoint selection, thresholds,
candidate contents and runtime margin. Upload only a fully verified candidate.
The official metadata stays byte-for-byte unchanged. The newest valid submission
counts; an upload is not a successful evaluated submission until CodaBench shows
valid status and metrics. No automatic CodaBench upload is built into this script.
