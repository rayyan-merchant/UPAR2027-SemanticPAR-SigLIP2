# UPAR2027 SemanticPAR — SigLIP2 NaFlex

Single selected architecture: SigLIP2-Base-NaFlex visual backbone, 40 pretrained
text-initialized attribute queries, masked patch cross-attention and a global
visual residual. Text inference and Hugging Face imports are preparation-only.

**See [STATUS.md](../STATUS.md) for actual execution evidence.** A code/test pass is
not a pretrained-parity pass, a trained model, an A10 timing result or a valid
CodaBench submission. No target leaderboard score is guaranteed.

## Audited inputs

Original files remain read-only under `D:/UPAR2027/official/data`.
97,669 official training and 33,407 validation records use the same ordered 40
attributes. All 131,076 annotated images were decoded and hashed. The official
PETA mapping resolves 15,631 missing canonical paths without moving originals.
One exact-pixel group spans train and two validation rows; official splits remain
unchanged, a sensitivity result excludes those validation rows, and calibration
does not use them. Near duplicates/shared identities are not fully assessed.

- [Implementation audit](../IMPLEMENTATION_AUDIT.md)
- [Data audit](../reports/data_mapped/DATA_AUDIT.md)
- [Kaggle CLI workflow](../KAGGLE_CLI.md)
- [Reproducibility record](../REPRODUCIBILITY.md)
- [Recipe](../configs/training.json)

## Execute on Kaggle

The user selected Kaggle CLI. The pipeline copies only official annotated data
to a **private** Kaggle source, re-audits it, runs tests, checks real pretrained
FP32/AMP parity, and benchmarks forward/backward batches before full training.
It keeps this architecture and effective batch size 64, reducing microbatch only
if needed for memory. See KAGGLE_CLI.md for data preparation and resume details.

```powershell
kaggle quota
python scripts/build_kaggle_kernel.py --owner rayyannmerchant --output C:/absolute/kernel
kaggle kernels push -p C:/absolute/kernel --accelerator NvidiaTeslaT4 -t 43200
kaggle kernels status rayyannmerchant/upar2027-siglip2-semanticpar
kaggle kernels output rayyannmerchant/upar2027-siglip2-semanticpar -p C:/absolute/run-output
```

The checked-in Kaggle runner executes one recipe: masked exponential-weighted
BCE, training-only prevalence, per-image `N_domain^-0.5` sampling, mild
augmentation, head-only warmup then upper four vision blocks, AdamW with layer
decay, warmup/cosine, AMP, clipping, accumulation and EMA. Raw and EMA checkpoints
are compared on validation. `last.pt` saves complete epoch-boundary training state;
`best.pt` is independently retained.

## Local verification

```powershell
python -m pip install -r requirements-training.txt
python -m pytest -q tests
python -m upar.inspect_data --train-csv D:/UPAR2027/official/data/annotations/task1/train/gt.csv --val-csv D:/UPAR2027/official/data/annotations/task1/val/gt.csv --data-root D:/UPAR2027/official/data --peta-mapping D:/UPAR2027/official/peta_file_mapping.txt --output reports/data_mapped
```

`upar.parity` accepts repeated `--image` arguments (at least three distinct real
aspect ratios). It compares installed HF preprocessing, embeddings, every visual
block, final features and text initialization. It reports maximum/mean absolute
and relative L2 error, and exports a hash-bound native foundation checkpoint only
after success. Full training also requires successful GPU AMP and feasibility
reports. The local helper `scripts/run_local_parity.py` uses this host's audited
images and performs CPU FP32 parity only.

## Evaluate, calibrate and package actual trained weights

```powershell
python -m upar.calibrate --predictions runs/siglip2/best_val_predictions.npz --output runs/siglip2/calibration.json
python -m upar.package --checkpoint runs/siglip2/best.pt --calibration runs/siglip2/calibration.json --output submission_candidate.zip
python -m upar.benchmark --submission-dir C:/absolute/unpacked-candidate --images-dir D:/UPAR2027/official/data/PA100k --n 2000 --batch-size 32
```

Global threshold offsets are subtracted from logits, making the evaluator's strict
0.5 cutoff implement calibrated decisions. Calibration estimates are in-sample,
not unbiased held-out results. No path/domain-specific threshold is used.

The inference ZIP contains root `run.py`, original `metadata.yaml`, native modules,
weights and calibration. Runtime uses torch, NumPy and Pillow only; no downloads,
Transformers, timm or text encoder. Both single-image and chunked batch APIs are
tested with network and optional model-library imports blocked.

## Evidence boundaries

CodaBench public rules specify Python 3.11/torch 2.4.1/CUDA 12.1, A10 24GB,
300 seconds total, 40 finite ordered probabilities, and equal per-dataset averaging.
The latest valid submission counts. Organizer scorer downloads are disabled:
published formulas are verified, but exact invalid/empty-class implementation
equivalence remains unresolved. Local metrics remain provisional.

The benchmark includes decode, preprocessing, model execution and JSON output,
and accounts for cold-start/warmup overhead. Any hypothetical workload projection
is explicitly labeled; Kaggle timing does not establish official A10 acceptance.

## Sources and data terms

- [Official challenge](https://www.codabench.org/competitions/18180/)
- [Official data and starter](https://github.com/speckean/UPAR-Challenge-2027)
- [Google SigLIP2 model](https://huggingface.co/google/siglip2-base-patch16-naflex)
- [SigLIP2 paper](https://arxiv.org/abs/2502.14786)

Use only official data for PAR optimization. General foundation pretraining is
permitted. Do not use hidden test images or metadata for training, adaptation,
selection or thresholds. Keep original dataset licences and citations: UPAR,
PA100K (HydraPlus-Net), Market1501 and PETA. Source annotations are CC BY-NC-SA 3.0;
image sources retain their own terms. Restricted images are never committed here.
