# Candidate A acceptance report

Candidate A is packaged and passes the completed correctness/data/licensing checks. It is ready for review of an **exploratory official evaluation**, with explicit upload approval now received. It is not certified to meet the240-second deployment target. The user explicitly directed that A10 timing be tested during submission; no local full29248-image GPU timing has been completed.

## Exact artifact

ZIP: `D:/UPAR2027/diagnostics/competitive-final/UPAR2027_Candidate_A_Frozen_DINOv3_75_25_v1.zip`

Size:650,810,045bytes. SHA256:`a54232287734b7e854ad472f1007c1830c081838121c4cd797bd343a686ac75d`.

The18-entry ZIP contains inference code, the original SigLIP checkpoint, frozen DINO backbone and trained head, separate ensemble calibration, unchanged official metadata, README and licensing notices. CRC passed. No feature cache, images, datasets, optimizer, history, private prediction arrays or training manifests are included. It is stored on D: because C: has insufficient safe headroom for another650MB artifact.

## Training and selection audit

All97,669 training rows and33,407 original validation rows match original annotation paths, labels, invalid masks, domain IDs and exact40-attribute ordering. Training CSV SHA256:`e42084aa43b31d4074624265b8239437afc6e47a359a8c0129fae4513a77d424`; validation CSV:`783be600e359052c9dafe856cbfa2aee45bbd4e3eac10309394e2388bcf7c2bc`.

A fixed seed-hash split of validation pixel groups gives16,744 selection and16,661 calibration images. Two validation images overlapping training pixels were excluded. Cross-partition pixel overlap is zero. Selection manifest SHA256:`8fea50be75aeb441f12344be24f30c58b8a7a770ce16f43dc40d9e7b01a1ca4b`; calibration manifest:`53aa464a6848f64632c60f028b3e91aee46338e09d49c42af6be0a6024f78dc8`.

The full backbone remained frozen; only training-row descriptors/labels entered head optimization and class-frequency computation. Verified cache/index/label hashes were reused. Thirty epochs completed in148.38seconds including setup/calibration. Epoch30 is the maximum recorded source-selection Challenge_Avg=.798357682. Best head SHA256:`6c2943f3cff1c0c2840e327ee6feea5aa7a1bf6d2f9ce0bf779e40c5aa5ec4f4`.

The curve was still improving: +.002268 from epoch25 to30, +.000228 from28 to30, with intervening fluctuations. A predeclared five-epoch source-only extension could be justified, but convergence is not established and no extension has been launched. Preserve epoch30 as the current candidate; do not extend in response to an official score.

## Calibration and combined inference

The ensemble is exactly .75*sigmoid(SigLIP logits)+.25*sigmoid(DINO logits), followed by one new source-calibration threshold vector mapped to the official0.5 output decision. No thresholded decisions or independently calibrated branch outputs are blended. Independent refitting on the16,661 calibration images reproduces the saved thresholds. The original production calibration remains unchanged.

Known-source selection Challenge_Avg is .867806 for Candidate A versus .865035 for the preserved production SigLIP calibration. Candidate A mA=.859955, instance-F1=.875970, precision=.850852, recall=.902730. These are development source measurements, not unseen-domain evidence: original SigLIP previously used the official validation set. Frozen LODO ensemble macro=.800964/worst=.780285 remains preserved and **failed its original.8023 promotion gate**. This official candidate is newly authorized as exploratory.

## Correctness and license tests

Nine focused A tests, two raw-mixture/preprocessing tests, and real-image probes passed. Under Python3.11.16/torch2.4.1 with network access blocked and no timm/Transformers/HF/safetensors deployment imports,3x40 valid probabilities were produced. Maximum discrepancy from saved AMP source predictions was.000906781.

On one byte-hash-verified original image from each domain, the independent timm1.0.24 DINO reference plus separately implemented aspect-fit normalization, overlapping-region pooling, attribute attention, mixture and calibration matched final probabilities within1.1921e-7. Preprocessing and all205 tokens matched exactly: CLS+4registers+200spatial tokens. Registers were excluded from spatial pooling. The SigLIP branch reuses the previously verified original native implementation and unchanged weights; no new independent trained HF SigLIP conversion is claimed.

DINOv3 and Apache license texts are included. README/NOTICE prominently say **Built with DINOv3**, identify Meta/timm/Google, and describe local modifications. The DINO model is the hash-pinned general-purpose timm port, not external PAR fine-tuned weights. UPAR permits general-purpose pretrained/foundation models; all PAR head training uses permitted data. Licensing sources: https://ai.meta.com/resources/models-and-libraries/dinov3-license/ ; https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m ; https://huggingface.co/google/siglip2-base-patch16-naflex .

## Runtime and official submission decision

Full29248-image GPU timing is **not measured locally**. The240-second target and300-second limit are unverified for this combined package. A bounded cold-process benchmark script includes imports/loading, decoding, both forwards, mixing, calibration and serialization; it does not substitute a T4 projection for A10 proof. As the user requested, the first A10/full official workload measurement is to occur during the exploratory CodaBench evaluation, with upload approval now granted. Do not report the deployment runtime gate as passed before that result.

Authenticated account rayyanmerchant showed0/3 daily and1/50 total submissions used;1003877 remains Finished. Live phase closes October22,2026 at12:00UTC/17:00PKT. Preferred upload13:00PKT. Organizer policy is **Force_Last**, latest rather than highest score. No previous-entry selection control was observed. Keep a submission slot for restoring the preserved original ZIP if needed; recheck quota immediately before upload. No new upload has been confirmed accepted; see the subsequent approved-upload status below.

Original accepted ZIP, best, last and calibration were freshly rehashed unchanged. B/C remain separate completed diagnostics under independent analysis; neither has started all-domain final training. Primary Kaggle quota last refreshed at7.78hours remaining. The second account has not been used for compute.

## Approved upload status

The user approved the exact sealed ZIP for an exploratory official A10 test. The upload reached 1% and reset before a new submission appeared. Last authenticated counters remained 0/3 daily and 1/50 total, with only 1003877 listed. Reconnection then failed with ERR_CONNECTION_TIMED_OUT. No new submission ID, A10 runtime or official score is confirmed. Approval remains valid for this unchanged artifact; the next attempt must first check for a delayed new entry to avoid duplication.

