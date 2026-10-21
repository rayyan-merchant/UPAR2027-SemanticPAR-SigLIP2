# Reproducibility and organizer verification

Architecture is fixed: Google SigLIP2-base-patch16-naflex visual tokens, 40
SigLIP2-text-initialized attribute queries, masked cross-attention, masked global
visual residual and independent logits. Text encoder is preparation-only.

Preserve the following with each real training run:

- Source commit and exact dependency freeze; GPU/CUDA/torch reports.
- Original annotation SHA256, official PETA mapping SHA256, decoded-image manifest,
  data audit, official attribute list and generated prompt list.
- Resolved foundation revision, prepared native checkpoint hash, FP32/AMP parity
  evidence with every encoder block's maximum/mean/relative L2 error.
- Training configuration, train-only prevalence, seed, full loss/validation
  history, raw/EMA comparisons, best checkpoint and complete resume checkpoint.
- Full official-validation metrics and the known-duplicate-excluded sensitivity
  result; per-attribute confusion/precision/recall/F1.
- Calibration input hash and 40 ordered thresholds. Calibration fit results are
  in-sample. Do not subsequently train on calibration labels and describe them as
  held-out estimates.
- Inference archive hash, original metadata comparison, offline API tests,
  measured cold-start/decode/preprocess/forward/serialization timings, hardware,
  and actual CodaBench valid-submission result when available.

Only official provided PAR training data is optimized. No hidden test images,
labels, metadata, domain names or structure may inform training or calibration.
The inference adapter uses `image_path` only to open pixels and validates the
provided ordered attribute vocabulary; it never selects offsets by path/domain.

The official public API pages were retrieved on 2026-10-19. Academic deadline is
Oct 21, 2026; CodaBench phase ends 2026-10-22 12:00 UTC (Oct 21 AoE). The CodaBench
announcement supersedes older workshop schedule text: code/models/fact sheet due
Oct 26 AoE; paper due Oct 23 AoE. Recheck official announcements before delivery.

Organizer code/model/fact-sheet delivery is required for ranking verification.
No email has been sent by this implementation. Do not include restricted images
in public source releases. The actual organizer scorer is not publicly exposed;
exact edge-case equivalence remains pending and must be stated in results.

Sources:
- https://www.codabench.org/competitions/18180/
- https://www.codabench.org/api/competitions/18180/
- https://github.com/speckean/UPAR-Challenge-2027
- https://huggingface.co/google/siglip2-base-patch16-naflex
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md


## Completed archive-based threefold diagnostics

Runtime e994492; private kernel upar2027-threefold-archives-resource-safe v1. See LODO_EVALUATION.md and reports/lodo/THREEFOLD_SUBMISSION_RECEIPT.json. Reproduce independent analysis with `python scripts/summarize_lodo.py --artifacts ../lodo/threefold-v2-results/lodo`, then `python scripts/finalize_lodo_report.py --repo . --artifacts ../lodo/threefold-v2-results/lodo`. The second command applies the reviewed recommendation; neither command trains or accesses competition data. Full private artifacts are required; weights remain server-locked and remote-only.


## Controlled equal-domain comparison completed

Runtime be2a8f0, private upar2027-lodo-equal-domain-comparison v1. Run `python scripts/compare_lodo_sampling.py --artifacts ../lodo/equal-domain-v1-results/lodo` with private diagnostic artifacts. It verifies the31file baseline lock, same source frequencies/BCE/config except sampling instrumentation, exact target/source scores and source-only threshold refits, plus measured exposure/unique-image bounds. Aggregate outputs use reports/lodo/equal_domain/results; never run default summarizer paths over locked baseline reports. The reviewed final report includes interpretation and the deadline plan. Retain original sampler; no improved production model or upload resulted.
