# Complete experiment and decision record

This supplement expands the three-page report. It records completed modeling experiments, access and infrastructure trials, rejected alternatives, final candidates and unexecuted suggestions. Only submission **1003877** has a confirmed official score. Local development scores are never substituted for an unobserved official result. An attempted follow-up upload was interrupted during the reported competition-site outage, and further planned official evaluation was not completed. A/B local comparisons remain evidence of expected improvement rather than unobserved official scores.

## 1 Choosing the task and protecting the data

We selected Track 1 pedestrian attribute recognition rather than adding detection, pose or segmentation. The task is 40 ordered binary attributes over cropped pedestrians; valid masks exclude unknown labels from optimization and evaluation. The source collection contains 97,669 training and 33,407 validation rows. A full image audit and official PETA mapping resolved file identities without changing original D: data. Two validation rows shared pixels with training, so later calibration and selection exclude them. Near-duplicate and person-identity leakage have not been exhaustively ruled out; exact-pixel disjointness is a narrower guarantee. Restricted images and private sample arrays are not included in the public code release.

Evidence: `reports/data_mapped/DATA_AUDIT.md`, `IMPLEMENTATION_AUDIT.md`, `reports/original_inventory.json` and `REPRODUCIBILITY.md`.

## 2 Building and submitting the original SigLIP2 system

We chose a general-purpose SigLIP2-Base-NaFlex foundation to combine visual features with pretrained semantic attribute queries. Native torch preprocessing and every vision block were checked against the reference, including GPU FP32/AMP, before PAR training. The model adds query-to-patch cross-attention and a global residual, produces 40 logits, and requires no text encoder at deployment. No unofficial PAR weights were used.

The executed recipe used head warmup then the top four vision blocks, AdamW, backbone/head rates 1e-5/1e-4, layer decay, warmup/cosine, EMA 0.9995, masked exponential class-balanced BCE, seed 2027, microbatch 16 with four accumulation steps and at most 256 patches. Per-image sampling weight is N_domain^-0.5, so total domain probability is proportional to sqrt(N_domain), not equal-domain. Frequencies come from unweighted training labels only. Twelve epochs completed; epoch-7 EMA won validation selection. The executed configuration in `reports/kaggle_v4/training/training_config.json` takes precedence over illustrative defaults.

Raw known-domain validation Challenge_Avg was 0.856185; duplicate-excluded raw was 0.856180; duplicate-excluded in-sample calibrated was 0.864952. Threshold offsets implement a global vector of 40 attribute decisions, not path/domain-specific rules. The same original validation set selected and calibrated production, so that score is optimistic. The archive passed offline API, checksum, attribute ordering and metadata checks. Submission **1003877** finished with Challenge_Avg **0.770450**, mA **0.773647**, instance-F1 **0.767279**, precision **0.741799**, recall **0.807915**, and label-F1 **0.583226**.

Original ZIP SHA256: `76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308`. Best checkpoint SHA256: `18a48956eac97d16709e355d57e747a9944c872f844e20645d1dc5aae86ae0d6`. Both remain private and preserved. Evidence: `TRAINING_RESULTS.md`, `reports/CODABENCH_SUBMISSION_RECEIPT.json`, `reports/kaggle_v4/`.

## 3 Investigating the validation gap rather than assuming calibration solved it

The calibrated source-to-official gap was about 9.45 percentage points. A disjoint calibration study achieved 0.863482, only about 0.147 percentage points below the corresponding fit estimate. This argued against calibration fitting alone as a sufficient explanation, but did not isolate checkpoint-selection optimism, domain shift or scorer conventions. We retained both harmonic-of-mean-image-P/R and mean-image-F1 conventions and unsupported-label sensitivities because the organizer's exact scorer is private. We did not download hidden images, labels or identities to explain the gap.

Evidence: `GENERALIZATION_DIAGNOSIS.md`, `scripts/generalization_analysis.py`, `reports/diagnosis_1003877/` where present. The primary local Challenge_Avg is the mean across datasets of the harmonic mean of mA and instance-F1; local instance-F1 is the harmonic mean of mean per-image precision and recall. It is not the arithmetic mean of mA and F1 and need not equal a metric computed from globally pooled counts.

## 4 Repairing staging before spending more GPU time

The first three-fold diagnostic staged mounted images individually. At the observed checkpoint it had copied 60,000 of 131,076 images after 35,703.7 seconds, with no training started. We confirmed this condition and cancelled the job; cancellation was later acknowledged. Linear projection was approximately 21.67 hours, not a completed measured staging time. High mounted-file overhead is supported by the comparison, while a specific provider backend fault is not proven.

We replaced repeated per-image mount access with deterministic private opaque ZIP archives, streamed/hash-verified as whole blobs and extracted to local working storage once. The initial PA-held union contained 52,073 images; a supplement added 79,003 missing files (79,001 PA training plus two PETA validation paths), giving the exact 131,076-image union for all folds. Full CPU staging passed in 34.4237 seconds, versus the earlier partial multi-hour loop. Source bytes and split hashes were unchanged. Staging 120-second, I/O 30-second and pretraining/stall watchdogs prevent extended idle GPU use. Independent PA/PETA workers were benchmarked before concurrent scheduling; Market followed on the freed GPU. No DDP was introduced for a fold.

Evidence: `STAGING_RECOVERY.md`, `reports/staging_recovery/`, `reports/lodo/THREEFOLD_CPU_STAGING.json`, `kaggle/archive_stage.py` and `kaggle/resource_policy.py`.

## 5 Establishing a fresh-foundation LODO baseline

Each held-out dataset was excluded from training, sampling, prevalence, checkpoint selection and calibration. Source training/selection/calibration and target evaluation were partitioned and hash-locked; the best source checkpoint and thresholds were fixed before one target evaluation. Folds started from the original general-purpose foundation, never the all-domain production model. Six epochs completed in all three folds; the suite took approximately 1.632 GPU-session hours with concurrent execution. Different source sizes (18,668, 89,001 and 87,669 training rows) and unequal update counts mean between-fold differences cannot be attributed only to domain difficulty.

Calibrated scores were PA100K 0.797104, PETA 0.777228 and Market1501 0.817448: macro 0.797260, worst 0.777228. Corresponding raw scores were 0.786128, 0.769342 and 0.813343. Source-only calibration helped each target but did not remove source-target gaps. Age, accessories and lower colors were recurring weak groups. All 40 source attributes had positives and negatives in every fold; PA-held lower-orange had only four positives, so sparse support matters without a blanket zero-positive explanation. This prompted controlled sampling and complementary-feature experiments rather than a large architecture search.

Evidence: `LODO_EVALUATION.md`, `reports/lodo/results/`, `scripts/summarize_lodo.py`.

## 6 Rejecting equal-domain sampling after a single controlled comparison

Only image sampling changed from N_domain^-0.5 to N_domain^-1, giving each source domain probability 0.5. Architecture, initialization, loss/BCE frequencies, augmentation, optimizer, six-epoch schedule, seed and sealed partitions stayed fixed. Measured draws and unique image coverage confirmed actual exposure; baseline exposure was not logged, so we state its expected probabilities rather than invent measured counts.

The calibrated macro fell from 0.797260 to 0.796089 and worst fell from 0.777228 to 0.776778. PA improved 0.000460, PETA regressed 0.000451, and Market regressed 0.003522. PETA gained mA but lost instance-F1; Market lost both. The +0.003 macro and worst-domain adoption margins failed. We retained square-root sampling and did not search another weighting. Reduced PA diversity versus more repetition of small domains is a plausible interpretation, not a demonstrated causal mechanism. Targets were already inspected, so this is development comparison evidence.

Evidence: `LODO_SAMPLING_COMPARISON.md`, `reports/lodo/equal_domain/results/COMPARISON.json`, `SAMPLING_EXPOSURE.csv` and `CONTROL_AUDIT.json` in the corresponding evidence directories.

## 7 Replacing an unavailable foundation route without using unknown weights

Direct access to the Facebook DINOv3 checkpoint was unavailable. We prepared the official DINOv2 ViT-B/14-with-registers fallback, then stopped its job when the user identified a maintainer-supported timm alternative. DINOv2 was not completed and has no reported model score. The public timm DINOv3 ViT-B/16 LVD-1689M checkpoint was legitimately accessible; we downloaded only safetensors and verified SHA256 `1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`. Meta DINOv3 and timm notices were retained. No account, geographic or explicit legal restriction was bypassed.

We implemented torch-only native inference and checked CLS, four registers, 200 spatial patches, FP32 RoPE, the reference QKV-bias convention and normalization against timm. CPU and GPU parity gates passed before extraction. A frozen-suite bootstrap first failed an inherited SigLIP preprocessing test because original reference dependency pins were omitted; restoring Transformers 4.57.1 and compatible tokenizers/hub fixed the reference environment without changing tests, tolerances or model methodology. The precise failed dependency version was not logged and is not claimed.

Evidence: `DINO_ARCHITECTURE.md`, `reports/dino_v3/GATE_PARITY.json`, `reports/dino_v3/FAILED_V1_RECOVERY.md` where present, `reports/dino_v3/failed_v1/`, `dino/v3_model.py` and `dino/licenses/`.

## 8 Frozen DINO heads and fixed complementary blending

An aspect-preserving 320x160 canvas feeds a frozen DINOv3 backbone. The six descriptors are CLS, spatial mean, head mean, upper-body mean, lower-body mean and spatial maximum. Overlapping regions avoid requiring exact pose. Each attribute learns attention over these six descriptors and an independent logit. We cached only compact FP16 descriptors, approximately 1.21 GB for all images, not full patch tensors. Each strict LODO head uses source-only training, original sampling/BCE, AdamW 3e-4, dropout 0.2, batch 128, seed 2027, up to 30 epochs with source-only stopping/selection. All three ran 30 epochs, with selected epochs 30/27/25. Descriptor extraction took about 250 seconds; heads took about 24/86/79 seconds.

DINO standalone calibrated scores were 0.754206/0.724861/0.783061, macro 0.754043 and worst 0.724861, so standalone replacement was rejected. Its probability residual correlations with SigLIP were 0.747-0.776 and binary-error correlations 0.537-0.558: useful incomplete agreement, not independent errors. The predeclared blend used 0.75 SigLIP raw sigmoid probabilities plus 0.25 DINO raw sigmoid probabilities, followed by a newly source-fitted global threshold vector. No mixing-weight search or held-out threshold fitting was performed.

Candidate A's ensemble scored 0.802375/0.780285/0.820231, macro 0.800964, worst 0.780285. Macro mA rose from 0.785674 to 0.786367; instance-F1 rose from 0.809226 to 0.816127. PETA's mA actually declined despite higher Challenge_Avg. Age and accessory balanced accuracy generally improved while clothing-color effects were mixed. The primary gate required macro >=0.8023, worst >=0.7772 and no domain regression >0.005. Macro failed by 0.001336; we preserved the failed gate but considered one near-useful limited fine-tuning experiment.

Evidence: `DINO_LODO_RESULTS.md`, `ENSEMBLE_COMPARISON.md`, `reports/dino_v3/results/`.

## 9 Candidate A all-domain head and the unsuccessful upload

Separate exploratory authorization led to a 30-epoch all-domain frozen head on 97,669 train rows, with 16,744 source-selection and 16,661 source-calibration rows after duplicate exclusions. Epoch 30 was selected and still improving; its head SHA256 is `6c2943f3cff1c0c2840e327ee6feea5aa7a1bf6d2f9ce0bf779e40c5aa5ec4f4`. A's calibrated source-selection ensemble score was 0.867806, not an unbiased estimate. The A offline package passed data, license, metadata and reference correctness checks. The user approved exploratory official evaluation.

The approved archive had 650,810,045 bytes and SHA256 `a54232287734b7e854ad472f1007c1830c081838121c4cd797bd343a686ac75d`. The upload reached 1%, then reset. Reconnection timed out, and the last observed table still showed only 1003877. The user reported the competition site was down and would upload later. No A ID or official score is confirmed. We did not duplicate the attempt. A prepared package is not a completed official experiment.

Evidence: `CANDIDATE_A_ACCEPTANCE.md`, `reports/competitive_final/CANDIDATE_A_*` receipts, especially `CANDIDATE_A_UPLOAD_RECEIPT.json`.

## 10 Candidate B source-controlled final-two-block adaptation

For each LODO fold, B started from the original DINOv3 foundation and that fold's source-trained frozen head. We did not initialize any held-out fold from an all-domain head. Only blocks 10/11, final normalization and head trained, with AdamW backbone/head rates 5e-6/5e-5, weight decay 0.01, original sampler and BCE, aspect-fit 320x160, horizontal flip p=0.5, microbatch 16/accumulation 4 and the same six-epoch comparison. Each target was evaluated after source selection and calibration; the fixed SigLIP fold and blend remained unchanged.

Standalone B calibrated macro was 0.783481, worst 0.752457. Its ensemble scored 0.803448/0.781356/0.822720, macro 0.802508, worst 0.781356. Macro mA 0.788772 improved on A's 0.786367; all three Challenge scores improved on A. Age and accessories benefited in some targets, but lower-color transfer remained mixed. Source-selected epochs were 6/5/5 after all six epochs completed. The suite took 4034.03 seconds, plus a 120-second failed bootstrap debit. No systematic multiple-factor or ensemble-weight search was made.

The branch-specific advance criterion required an additional +0.002 macro over A, i.e. >=0.802964, not merely the older absolute 0.8023 threshold. B improved +0.001544 and missed by about 0.000456. We recorded a failed advancement gate. Later exploratory final-training authorization was a separate decision; we did not retroactively convert this negative result into promotion. Exact private checkpoint bytes for these LODO folds were verified remotely, not all rehashed locally; the final all-domain B best checkpoint was subsequently independently downloaded and rehashed.

Evidence: `DINO_FINETUNE_LODO.md`, `reports/competitive_final/candidate_b/SUMMARY.json` and `SOURCE_CURVES.json`.

## 11 Candidate C isolated surveillance augmentation

This independent branch retained the original SigLIP foundation, architecture, loss, sampling and training schedule. Only augmentation changed: bounded down/up sampling p=0.30, scale 0.65-1; gamma p=0.20, 0.9-1.1; brightness p=0.50, 0.9-1.1; JPEG p=0.30, quality 60-95; blur p=0.15, sigma 0.2-0.7, alongside the existing augmentation. Each LODO fold started fresh without production PAR weights.

All three six-epoch folds completed. Calibrated scores were 0.797197/0.776418/0.820613, macro 0.798076 (+0.000816) and worst 0.776418 (-0.000810). The +0.003 macro/worst requirements failed; a small macro gain could not justify making the weakest target worse. The suite took 6002.41 seconds plus a 60-second bootstrap debit. We rejected C permanently and did not train, package or submit an all-domain C model.

Evidence: `SIGLIP_ROBUSTNESS_ABLATION.md`, `reports/competitive_final/candidate_c/`.

## 12 Final exploratory B training and technical recoveries

The user separately authorized one all-domain B run. This time, using A's all-domain head is legitimate because there is no held-out LODO domain: all 97,669 official training images are permitted. Backbone initialization remains the original pretrained DINOv3. The train/selection/calibration split, weights, masks and annotation identities are sealed; only blocks 10/11, final norm and head are optimized. The original SigLIP model remains unchanged for deployment blending. No EMA was introduced to the DINO fine-tuning recipe.

Final-B v1 passed 25 tests but hit the 600.31-second pretraining watchdog. A new annotation audit repeatedly decompressed NPZ members per row. Loading arrays once and vectorizing the same checks reduced the complete 131,076-row local audit to 28.44 seconds, without image reads. V2 passed 27 tests, then failed before staging/backward at a locator that pruned the `runs/` directory containing pinned SigLIP predictions. A scoped locator included runs/ and still required the exact original prediction SHA256. No model trained in either failed attempt; their approximately 0.20 GPU-hour cost was charged, not reset. V3 passed 28 cloud tests and completed.

Six reference epochs were followed by at most two source-only extensions. Epoch 6 exceeded all prior raw DINO source scores by >=0.0001, permitting epoch 7; epoch 7 repeated that condition, permitting epoch 8. Scores 6/7/8 were 0.838134/0.839221/0.840005, with shrinking gains. Epoch 8 was selected. Total script time was 5026.07 seconds, with 12,216 attempted/12,214 successful updates and two AMP skips. There were 781,352 draws: PA 462,524, PETA 153,861 and Market 164,967. Unique images seen were 78,749/8,668/10,000; replacement sampling did not guarantee every PA image was visited. The source backward probe measured 198.68 images/s and 0.763 GiB peak allocated memory.

Final B's source-selection ensemble Challenge_Avg was 0.869305, mA 0.860692, instance-F1 0.878215, precision 0.853342 and recall 0.904729. That is separate from B's LODO macro 0.802508. Best checkpoint SHA256 `6e9fab596868099b1b8915349b5c9e935693c3a56836ea229c94037af7761c20` was downloaded with resumable chunks and independently verified. The original foundation, A head, source splits, new calibration and raw prediction alignment were checked. An additional two-epoch continuation was discussed but never launched; its approximately 20-minute training/validation cost was an estimate, not a completed experiment.

Evidence: `FINAL_B_EXECUTION.md`, `FINAL_B_RECOVERY.md`, `FINAL_B_TRAINING_REPORT.md`, `reports/competitive_final/final_b_failed_v1/`, `final_b_failed_v2/` and `final_b_results/`.

## 13 Verifying that the final models implement the evaluated method

The B wrapper handles CLS plus four registers and 200 spatial patches, pools the six descriptors, blends underlying raw branch probabilities and applies one new source-only 40-threshold vector. It preserves FP32 descriptor pooling used by fine-tuning; A's frozen-cache FP16 round-trip remains confined to A. We verified 134 frozen backbone tensors exactly match original foundation values. Independent timm inference on one real image per source domain gave identical preprocessing and spatial tokens, and final probability maximum absolute error 1.1921e-7. Five wrapper tests passed. These checks connect the experimental method to the actual model implementation, rather than relying only on training metrics.

The original SigLIP checkpoint and calibration, A head, final B best checkpoint, split manifests and new source-only ensemble calibration all have retained SHA256 receipts. The final A/B packages exclude feature caches, optimizer/history, training images and sample-level arrays, and preserve original metadata and model notices. Technical receipts remain in the research archive; the assignment narrative concentrates on method and local evaluation.

Evidence: reference, checkpoint, source-calibration and package receipts in `reports/competitive_final/final_b_results/`; native code in `dino/` and `upar/`.

## 14 What was selected and what an unmade submission could have scored

The only confirmed official entry and final evaluated fallback is original SigLIP2 **1003877**, score **0.770450**. A is a prepared unscored frozen ensemble; B is a prepared unscored partially fine-tuned ensemble; C is rejected. B is the strongest measured local transfer candidate, but neither its local macro 0.802508 nor any source-validation score can be reported as its hypothetical official score. The original system's source/official gap demonstrates why adding local gains to 0.770450 would be unjustified. Both ensembles improved every local held-out domain, which supported our expectation of better generalization and made them promising candidates to improve the official result. Competition-site unavailability interrupted the attempted follow-up evaluation, so that expectation remains unconfirmed; no exact official score can be assigned to an unscored model.

The previously verified competition policy was Force_Last: newest eligible submission, not automatically best. The last authenticated allowance observed before A's failed attempt was 0/3 daily and 1/50 total. These are historical observations, not a current live allowance. A latest-entry replacement carries selection risk, so original ZIP/checkpoints/calibration were retained for rollback; manual previous-entry selection was not confirmed. The verified deadline record was October 22, 12:00 UTC / 17:00 PKT. No second account was used to circumvent platform or submission limits. This report makes no current claim that the site remains down or that a new entry exists.

## 15 Limits and the next improvement worth doing

Repeatedly inspected LODO targets are development data; one seed, unequal source sizes/update counts, sparse attributes and incomplete convergence limit inference. Generalization gains cannot identify a single cause such as representation degradation. A helped mainly instance-level precision/recall; B added modest mA gains; C and equal sampling show that plausible changes can trade away worst-domain transfer. Exact organizer F1/unsupported-label conventions remain uncertain, so both primary and sensitivity scores are retained.

The next scientific step is repeated-seed and matched-compute validation of B's conservative adaptation. Source-selected continuation could test remaining optimization headroom without changing the backbone or hidden-test thresholds, while monitoring whether additional source learning preserves worst-domain transfer. Broader rare-attribute support and source-only calibration-transfer analysis are plausible research directions, not implemented wins. The stretch official target was an aspiration, never a claimed result.

## Reading the evidence without obsolete status confusion

`STATUS.md` and the assignment README are the current summary. Original research reports and hash-locked files are retained byte-for-byte where required: their earlier statements such as 'not launched' describe the stage at which they were written. This supplement and final receipts explicitly supersede those historical pending statements. We preserve negative gates and technical failures because deleting them would distort the experimental story. Generated caches, local environments, private weights/data and transient bundles are ignored and excluded from the assignment source archive.
