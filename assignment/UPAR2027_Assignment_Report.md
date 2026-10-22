# Pedestrian attribute recognition under domain shift
UPAR 2027 Track 1 assignment report  |  22 October 2026

## From a working classifier to a generalization problem

We selected UPAR 2027 Track 1, a 40-label pedestrian attribute-recognition task covering age, gender, hair, clothing and accessories. Our goal was to learn attribute features that transfer across camera datasets rather than merely fit familiar images. We completed several controlled experiments, but only the original SigLIP2 model has a confirmed official evaluation: submission 1003877, Challenge_Avg 0.770450. The later ensembles have stronger local transfer results; further official evaluation was disrupted by competition-site unavailability.

The permitted data contained 97,669 training and 33,407 validation rows from PA100K, PETA and Market1501. We kept originals read-only, resolved PETA filenames through the official mapping, audited all 131,076 images and preserved the exact 40-attribute order. One exact-pixel group linked a training image to two validation rows. Those validation rows were excluded from later selection and calibration. Unknown labels were masked rather than treated as negatives; images, private predictions and weights were kept outside the code repository.

## The first model and what its official score taught us

Our initial model used SigLIP2-Base-NaFlex, aspect-aware inputs of up to 256 patches, 40 text-initialized attribute queries, masked patch attention and a global visual residual. We chose semantic initialization to reuse general-purpose visual-language knowledge without requiring a text encoder at deployment. Training used masked class-balanced BCE, training-only label frequencies, per-image weights proportional to inverse square-root domain size, AdamW, head warmup, upper-four-block adaptation, cosine scheduling and EMA. Twelve epochs completed; validation selected the epoch-7 EMA checkpoint.

Known-domain validation Challenge_Avg was 0.856185 at threshold 0.5 and 0.864952 after in-sample calibration with duplicate rows excluded. However, official Challenge_Avg was 0.770450, with mA 0.773647, instance-F1 0.767279, precision 0.741799 and recall 0.807915. This gap showed why source validation alone was insufficient. A disjoint calibration audit gave 0.863482, suggesting calibration-fit optimism could not by itself explain the gap. We therefore investigated domain transfer rather than repeatedly adjusting thresholds on the same familiar data.

## Designing an evaluation that tests the real weakness

We introduced three leave-one-domain-out (LODO) folds: train on two datasets and test on the third. This directly tests whether the learned features survive a change of cameras, appearance distributions and annotation balance. Source-training, checkpoint-selection and calibration partitions were disjoint. Fresh foundation initialization, source-only class weights and locked checkpoint/threshold hashes preceded one target evaluation per recipe. Six epochs per fold provided a fixed diagnostic budget, not proof of convergence. A slow initial data-copy workflow was replaced by verified private archives extracted once and reused across folds, allowing us to complete the controlled comparisons without changing image bytes or splits.

<!-- Page break in the PDF -->

# Controlled experiments and the decisions they supported
One factor per branch  |  Source-only checkpoint selection and calibration

## Testing sampling before changing the backbone

The SigLIP2 LODO baseline scored 0.797260 macro and 0.777228 worst-domain after source-only calibration. PETA was weakest; age, accessories and lower colors remained difficult. All source attributes had positive and negative examples, but lower-orange had only four positives in the PA-held fold. Sparse support mattered without explaining every transfer failure.

We tested whether PA100K dominated learning by changing only sampling to 50/50 source-domain probabilities. Macro fell to 0.796089, worst to 0.776778, and Market1501 lost 0.003522. Draw logs confirmed the change with BCE frequencies held fixed. We retained square-root sampling: equal exposure was plausible but unsupported by results. Previously inspected targets make these development comparisons rather than unbiased certification.

## A complementary representation and a conservative ensemble

We next tested self-supervised DINOv3 ViT-B/16 features to complement SigLIP2's language-aligned representation. After direct access was unavailable, a maintainer-supported timm checkpoint passed provenance/license checks; the uncompleted DINOv2 fallback was cancelled. At 320x160, DINOv3 produces CLS, four registers and 200 spatial patches. Six descriptors combine CLS, spatial mean, overlapping head/upper/lower means and spatial maximum. Attribute-specific attention learns where each label's evidence is located without a pose model.

We froze the backbone to preserve general-purpose features and trained source-only heads for up to 30 epochs on cached FP16 descriptors. DINO alone was weaker, but its errors partly differed from SigLIP2. A predeclared 75/25 raw-probability mixture, with new source-only calibration, improved every fold to macro 0.800964 and worst 0.780285. Candidate A missed its 0.8023 gate. Gains mainly came from instance precision/recall; macro mA rose from 0.785674 to 0.786367.

## Measured calibrated transfer

| Method | PA100K | PETA | Market | Macro | Worst |
|---|---:|---:|---:|---:|---:|
| SigLIP2 baseline | .797104 | .777228 | .817448 | .797260 | .777228 |
| Equal sampling | .797565 | .776778 | .813926 | .796089 | .776778 |
| Frozen DINO alone | .754206 | .724861 | .783061 | .754043 | .724861 |
| A frozen ensemble | .802375 | .780285 | .820231 | .800964 | .780285 |
| B adapted ensemble | .803448 | .781356 | .822720 | .802508 | .781356 |
| C augmentation | .797197 | .776418 | .820613 | .798076 | .776418 |

Local Challenge_Avg is the per-domain harmonic mean of mA and instance-F1, then averaged equally over domains. These are development scores, not official scores.

## Two independent follow-up branches

Candidate B adapted only DINO blocks 10/11, final normalization and head, preserving earlier features. Backbone/head rates were 5e-6/5e-5 for six epochs. Its ensemble reached macro 0.802508, worst 0.781356 and mA 0.788772, but the +0.001544 gain over A missed its +0.002 margin. Candidate C independently added bounded JPEG, resizing, blur, brightness and gamma to SigLIP2 training. Macro 0.798076 concealed a PETA regression to 0.776418, so we rejected this augmentation policy.

<!-- Page break in the PDF -->

# Final candidates and what we can conclude
Experimental interpretation, final selection and future improvements

## From complementary features to final models

The results favored complementarity over replacement: frozen DINO alone scored macro 0.754043, and partial adaptation raised it to 0.783481, both below SigLIP2's 0.797260. Their fixed ensembles nevertheless reached 0.800964 and 0.802508. A weaker standalone model can help when its errors differ. We fixed the blend in advance and retained failed advancement margins rather than redefining success after seeing results.

Although gains missed our conservative margins, improvements across all folds motivated an exploratory final comparison. A/B trained on all 97,669 official training images, with 16,744 selection and 16,661 calibration rows. A's head was selected at epoch 30. B initialized the original DINOv3 backbone and A's all-domain head; only its final two blocks, normalization and head trained. Source improvements justified the two predeclared extensions; all eight epochs completed and epoch 8 was selected. Final ensemble source-selection scores were A 0.867806 and B 0.869305; these are development results, separate from LODO evidence.

## What we submitted and expected to improve

We submitted the original SigLIP2 system as entry 1003877, receiving official Challenge_Avg 0.770450. We then completed and prepared the stronger A/B ensembles for further evaluation, but the competition site became unavailable. A's upload reset and reconnection failed, leaving no confirmed new ID; further planned official evaluation was not completed. Original SigLIP2 therefore remains our only officially scored model, while the later models have completed local evaluations.

Both ensembles improved all three held-out domains. B gained 0.005248 macro (0.525 percentage points) and 0.004127 worst-domain over baseline, while mA rose from 0.785674 to 0.788772. We consequently expected better generalization from the later ensembles. This is evidence supporting an improved official result, not a measured hidden score: the exact result had they been submitted is unknown. We report the demonstrated local gains without substituting them for unobserved official results.

## Challenges and improvements supported by the experiments

The central challenges were the source-to-official gap, unequal dataset sizes, sparse attributes, reference-model access and leakage prevention. Independent checks reproduced DINO preprocessing and spatial tokens exactly; final probabilities differed from the reference by at most 1.19e-7. Source-only selection/calibration and separate fold initialization protected the comparisons. However, one seed, unequal update counts, repeated target inspection and unresolved scorer edge cases limit causal and statistical conclusions. Age and accessories benefited in several settings, while lower colors remained inconsistent.

Next steps are repeated-seed and matched-compute comparisons, better rare-attribute coverage and calibration-transfer analysis. A short source-selected continuation could test B's remaining learning potential because epoch 8 was still improving; it was discussed but not run. More adaptation is not automatically better, so a frozen reference and worst-domain mA should guide further work. The assignment contributes one official submission and a connected record of positive and negative experiments identifying the complementary ensemble as the most promising next model.

## References and reproducibility

Official task: https://github.com/speckean/UPAR-Challenge-2027 ; competition: https://www.codabench.org/competitions/18180/ ; foundations: https://huggingface.co/google/siglip2-base-patch16-naflex and https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m . Detailed evidence is in assignment/EXPERIMENTS_AND_DECISIONS.md, assignment/RESULTS.csv and docs/EXPERIMENT_INDEX.md.