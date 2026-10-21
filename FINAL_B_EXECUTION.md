# Final Candidate B execution

User separately authorized one exploratory all-domain B run after reviewing the completed LODO results. B's predeclared research advancement gate remains FAILED: macro .802508 versus required .802964. This authorization does not change that historical result. C is rejected and will not run further. Original1003877 and Candidate A remain independently preserved.

## Fixed recipe and provenance

Use all97,669 official training images, verified original DINOv3 ViT-B/16 foundation SHA1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332, plus A's epoch30 frozen regional head SHA6c2943f3cff1c0c2840e327ee6feea5aa7a1bf6d2f9ce0bf779e40c5aa5ec4f4. Only blocks10/11, final normalization and head train. AdamW backbone5e-6/head5e-5, wd.01, masked balanced BCE from unweighted training prevalence, square-root-domain sampler, seed2027, microbatch16/accum4, dropout.2, flip.5, aspect-fit320x160, original40 attributes. No full-backbone fine-tune, no LODO checkpoint or all-domain PAR backbone initialization.

Reuse A's sealed pixel-disjoint source selection16,744/calibration16,661 partitions. Exclude the two validation pixels duplicated in training. Check original annotation, index, label, A checkpoint/config/partition/manifest hashes and each row's labels/masks/domain. Class weights use training only. No hidden-test statistics or labels.

Six reference epochs are mandatory. Before seeing new results, define optional extension: epoch7 allowed only when epoch6 beats every earlier source-selection score by>=.0001; epoch8 allowed only when epoch7 again beats all earlier scores by>=.0001. Stop by epoch8. Choose best raw DINO source-selection checkpoint only. This epoch plan remains as the user specified; the user subsequently removed the proposed arbitrary two-hour session cap. No artificial training wall-time cap or Kaggle timeout override is configured. Platform quota and deadline limits still apply.

## Resource gates and recovery

Primary quota freshly checked7.78h remaining. Stage four already verified private archives once,120s maximum/30s I/O stall; no mounted individual-image copy/hash/decode loop. Reuse extracted data for all epochs. DataLoader30s timeout, separate logs, training progress/ETA and180s no-progress infrastructure watchdog; pretraining setup600s. Only GPU0 is assigned to this single independent all-domain run; no unverified DDP or duplicate model on GPU1. Before training measure51 source backward microbatches, finite gradients, exact frozen/trainable parameters, actual memory and throughput. Gate at>=60images/s and<=12GiB allocated. Project worst8epoch cost with35% training margin and setup/evaluation allowance; require it fit the fresh quota while reserving2h for deployment/recovery. No training begins on a resource conflict.

Best and atomic last checkpoints preserve optimizer/scaler/Python/NumPy/torch/CUDA/sampler RNG, source-domain draws, unique exposure and history. No EMA is introduced because the validated B recipe did not use it. Transient extracted images/archive cache are removed after successful completion; best/last and all audit evidence remain privately in Kaggle outputs.

## Ensemble and deployment

Use the pinned original SigLIP raw source-validation predictions and exactly75%SigLIP raw sigmoid +25%B raw sigmoid. Fit one40-attribute threshold vector using only the calibration partition after checkpoint selection. Save independent source metrics, predictions, hashes and locks. Do not reuse A's ensemble thresholds. Original SigLIP weights/calibration remain untouched.

Training job performs no competition upload. After actual completion independently audit metrics/calibration/locks, assemble torch-only offline package, verify against both references on real images, validate metadata/licenses/40-order/API, and measure inclusive decoding/load/both-forwards/mixing/calibration/serialization runtime. Target<=240s, require official300s contract; no A10 runtime pass is claimed in advance. Official evaluation is authorized only after deployment gates and a fresh submission-policy/allowance check. CodaBench was last unreachable; verify current access before action. Force_Last means latest, not automatically best; manual prior-entry selection remains unverified.

Deadline verified previously: October22,2026 12:00UTC/17:00PKT; preferred upload13:00PKT. No score improvement is guaranteed.

## Launch preparation

Private kernel prepared: rayyannmerchant/upar2027-candidate-b-final-all-domain. Script SHA2561d556df17b058c043a9aeaf5bfc27b39436d7ca9011b9b2ab91797ac2de414ba. Six focused local tests passed. Actual cloud gate, training state and results must be read from saved evidence; preparation is not training completion.

Private version1 submitted once and verified QUEUED at2026-10-21T15:44UTC. Source85b2c61. No cloud throughput/gate/epoch result is claimed yet. Existing follow-up updated for final B and active; it must stay quiet on unchanged state. Provider timeout override omitted as requested.


V1 failed before staging/training after25tests passed. The annotation audit repeatedly decompressed NPZ arrays per row; see FINAL_B_RECOVERY.md. Fixed full131076-row audit passed locally in28.44s;8focusedtests pass. Remainingquota7.61h; planned v2 preserves the method and all original/A artifacts. No training time cap added.

