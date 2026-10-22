# Pedestrian attribute recognition under domain shift
UPAR 2027 Track 1 assignment report  |  22 October 2026

## From a working classifier to a generalization problem

We selected UPAR 2027 Track 1, a 40-label pedestrian attribute-recognition task covering age, gender, hair, clothing and accessories. Our goal was a reproducible model that could transfer between camera datasets and run offline within the competition's 300-second inference limit. We completed several controlled experiments, but only the original SigLIP2 model has a confirmed official evaluation: submission 1003877, Challenge_Avg 0.770450. Later candidates have local results, not official scores.

The permitted data contained 97,669 training and 33,407 validation rows from PA100K, PETA and Market1501. We kept originals read-only, resolved PETA filenames through the official mapping, audited all 131,076 images and preserved the exact 40-attribute order. One exact-pixel group linked a training image to two validation rows. Those validation rows were excluded from later selection and calibration. Unknown labels were masked rather than treated as negatives; images, private predictions and weights were kept outside the public repository.

## The first model and what its official score taught us

Our initial model used SigLIP2-Base-NaFlex, aspect-aware inputs of up to 256 patches, 40 text-initialized attribute queries, masked patch attention and a global visual residual. We chose semantic initialization to reuse general-purpose visual-language knowledge without requiring a text encoder at deployment. Training used masked class-balanced BCE, training-only label frequencies, per-image weights proportional to inverse square-root domain size, AdamW, head warmup, upper-four-block adaptation, cosine scheduling and EMA. Twelve epochs completed; validation selected the epoch-7 EMA checkpoint.

Known-domain validation Challenge_Avg was 0.856185 at threshold 0.5 and 0.864952 after in-sample calibration with duplicate rows excluded. However, official Challenge_Avg was 0.770450, with mA 0.773647, instance-F1 0.767279, precision 0.741799 and recall 0.807915. The official log recorded 29,248 images in 109.91 seconds. This gap showed why source validation alone was insufficient. A disjoint calibration audit gave 0.863482, suggesting calibration-fit optimism could not by itself explain the gap; domain shift, selection effects and scorer differences remained confounded.

## Replacing a slow workflow with a controlled evaluation

We therefore introduced three leave-one-domain-out (LODO) folds. Each model trained on two source domains; source-training, checkpoint-selection and calibration partitions were disjoint. Fresh foundation initialization, source-only class weights and locked checkpoint/threshold hashes preceded one target evaluation per recipe. Six epochs per fold provided a fixed diagnostic budget, not proof of convergence. An initial staging design stalled at 60,000/131,076 mounted files after 35,703.7 seconds, before training. We cancelled it, packaged verified images into four private archives and benchmarked full staging at 34.42 seconds. Local extraction was reused across folds, with staging and I/O watchdogs and independent two-GPU scheduling.

<!-- Page break in the PDF -->

# Controlled experiments and the decisions they supported
One factor per branch  |  Source-only checkpoint selection and calibration

## Testing sampling before changing the backbone

The completed SigLIP2 LODO baseline scored 0.797260 macro and 0.777228 worst-domain after source-only calibration. PETA was the weakest target; age, accessories and lower colors remained difficult. Every source attribute had positive and negative examples, although lower-orange had only four positives in the PA100K-held-out fold. We therefore rejected a blanket explanation based on missing source-positive labels.

A single equal-domain experiment changed only sampling from square-root weighting to 50/50 source-domain probabilities. Macro fell to 0.796089 and worst-domain to 0.776778; Market1501 lost 0.003522. Logged draws confirmed the requested exposure and unchanged BCE frequencies. We retained the original sampler. This was a predeclared development comparison on previously inspected targets, not a fresh unbiased test.

## A complementary representation and a conservative ensemble

We next tested DINOv3 ViT-B/16 as a complementary visual representation. Direct Facebook access was unavailable; a DINOv2-with-registers fallback was prepared and its job cancelled when a maintainer-supported public timm DINOv3 route passed access, license and SHA256 checks. DINOv2 has no completed comparative result. The DINOv3 model used an aspect-preserving 320x160 canvas, CLS plus four registers and 200 spatial patches. Six descriptors combined CLS, spatial mean, overlapping head/upper/lower regions and spatial maximum; attribute-specific attention produced 40 logits.

We froze the backbone, cached compact FP16 descriptors once, and trained source-only heads for up to 30 epochs. DINO alone was weaker, but its errors were incompletely correlated with SigLIP2. We predeclared a 75% SigLIP2 / 25% DINO raw-probability mixture and fitted a new 40-threshold vector on source calibration only. Candidate A improved all three folds to macro 0.800964 and worst 0.780285, but missed its 0.8023 promotion threshold. Most gain came from instance precision/recall; macro mA rose only from 0.785674 to 0.786367.

## Measured calibrated transfer

| Method | PA100K | PETA | Market | Macro | Worst |
|---|---:|---:|---:|---:|---:|
| SigLIP2 baseline | .797104 | .777228 | .817448 | .797260 | .777228 |
| Equal sampling | .797565 | .776778 | .813926 | .796089 | .776778 |
| Frozen DINO alone | .754206 | .724861 | .783061 | .754043 | .724861 |
| A frozen ensemble | .802375 | .780285 | .820231 | .800964 | .780285 |
| B adapted ensemble | .803448 | .781356 | .822720 | .802508 | .781356 |
| C augmentation | .797197 | .776418 | .820613 | .798076 | .776418 |

Local Challenge_Avg is the per-domain harmonic mean of mA (label-balanced accuracy) and instance-F1, then averaged equally over domains. Local F1 uses harmonic mean of mean-image precision/recall; exact organizer edge cases remain unresolved. No row is an official hidden-test score.

## Two independent follow-up branches

Candidate B adapted only DINO blocks 10/11, final normalization and the source-trained head, using backbone/head rates 5e-6/5e-5 and six epochs. The fixed ensemble reached macro 0.802508, worst 0.781356 and macro mA 0.788772, improving all domains over A. Nevertheless, its +0.001544 macro gain missed the predeclared +0.002 margin; we preserved that failed advancement decision. Candidate C changed only SigLIP2 augmentation with bounded JPEG, resizing, blur, brightness and gamma degradation. Its macro 0.798076 hid a PETA worst-domain regression to 0.776418, so we rejected C and did not train an all-domain C model.

<!-- Page break in the PDF -->

# Final candidates and what we can conclude
Packaging, measured limitations and the final submission decision

## Completing exploratory candidates without rewriting the research gates

Separate exploratory authorization allowed an all-domain frozen A head, selected at epoch 30, and a final B run despite the failed research gates. B initialized the verified original DINOv3 foundation and A's all-domain head, with only the final two blocks, normalization and head trainable. It reused 97,669 training, 16,744 selection and 16,661 calibration rows. Epochs 6 and 7 each improved the source-selection score enough to justify the two predeclared extension epochs; all eight completed and epoch 8 was selected. Raw DINO selection rose from 0.838134 at epoch 6 to 0.840005 at epoch 8. We did not launch further training.

Final B took 83.77 minutes, with 12,216 attempted updates, 12,214 successful updates and two AMP skips. Its calibrated ensemble source-selection Challenge_Avg was 0.869305, versus A's 0.867806 on the same partition. These source results do not replace LODO evidence or predict the hidden score. Two pre-training B failures exposed repeated compressed-array decompression and a prediction-file locator excluding runs/. Both were corrected with unchanged identity checks and pinned hashes; no failed attempt had trained a model.

## Correctness passed while deployment timing remained unresolved

We preserved the original production checkpoint/calibration and exported separate offline A and B packages. The mixture always combines raw sigmoid probabilities before the new source-only calibration. B's independent timm check reproduced preprocessing and spatial tokens exactly on real images; final probabilities differed by at most 1.19e-7. Offline Python 3.11 / torch 2.4.1 loading passed without network or optional model libraries. All 134 frozen backbone tensors matched the original foundation. The B ZIP passed CRC checks and contained only inference code, weights, calibration, original metadata and license notices, excluding training images, features and optimizer state.

The exact B payload's full-workload T4 benchmark timed out at 300 seconds; last saved progress was 26,624 of 29,248 source images at 296.717 seconds. Staging took 9.115 seconds, isolating inference as the immediate bottleneck. A10 timing is unknown; a T4 timeout neither proves an A10 failure nor certifies success. Candidate A's approved upload reset at 1% and reconnection timed out during the reported competition-site outage, producing no confirmed ID. B also remained blocked by its deployment gate. Consequently only submission 1003877 has a confirmed official score; the outage is not presented as the sole reason B was unsubmitted.

## Final selection and improvements supported by the evidence

The final officially evaluated model remains the original SigLIP2 submission at 0.770450. B is the strongest measured local transfer candidate, but neither its 0.802508 LODO macro nor its 0.869305 source score is a hypothetical official score. Had A or B been submitted, the outcome would still be unknown until successful official inference and scoring. We cannot claim that the 0.8021 stretch target would have been reached.

Next work should first profile and accelerate the complete inference path while preserving predictions, then verify the actual A10 runtime. A small source-selected B continuation is scientifically plausible because epoch 8 was improving, but diminishing gains and unresolved runtime make it secondary. Repeated seeds, stronger rare-attribute support, calibration-transfer studies and matched-compute LODO comparisons would better separate sampling, representation and dataset-size effects. We preserved failed gates and negative results rather than choosing a recipe from hidden scores. The repository includes working code, a machine-readable result table and a separate detailed experiment/decision record.

## References and reproducibility

Official task and data: https://github.com/speckean/UPAR-Challenge-2027 and https://www.codabench.org/competitions/18180/. Foundations: https://huggingface.co/google/siglip2-base-patch16-naflex and https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m. Implementation and complete evidence: repository README, assignment/EXPERIMENTS_AND_DECISIONS.md and assignment/RESULTS.csv. All numeric claims derive from retained reports; exact source paths are recorded in the experiment ledger.
