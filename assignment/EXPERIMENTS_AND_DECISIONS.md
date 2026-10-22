# Experiments and decisions

This supplement records the scientific methods, intuition and conclusions behind the methodology and results report. Internal job receipts, debugging logs, account helpers and execution plans are not part of the assignment hand-in. Measured local values are published in `RESULTS.csv`; only original submission 1003877 has a confirmed official score.

## 1 Task and original model

UPAR Track 1 predicts 40 ordered pedestrian attributes from crops. We used 97,669 official training and 33,407 validation rows across PA100K, PETA and Market1501. Unknown labels are masked. PETA filenames follow the official mapping, originals remain read-only, and two validation rows with training-duplicate pixels are excluded from later selection/calibration. Near-duplicate/person-identity overlap was not exhaustively resolved.

The original SigLIP2-Base-NaFlex model combines text-initialized attribute queries, masked patch attention and a global visual residual. Semantic initialization reuses general-purpose visual-language knowledge without a text encoder at inference. Training used masked balanced BCE, training-only class frequencies, square-root-domain sampling, AdamW, head warmup, upper-four-block adaptation, cosine scheduling, EMA and seed 2027. Twelve epochs completed; validation selected the epoch-7 EMA checkpoint.

Raw known-domain validation Challenge_Avg was 0.856185 and duplicate-excluded in-sample calibrated performance was 0.864952. Official submission **1003877** finished at **0.770450**, mA **0.773647**, instance-F1 **0.767279**, precision **0.741799** and recall **0.807915**. A disjoint calibration analysis scored 0.863482, suggesting calibration fitting alone did not explain the source/official gap. This motivated evaluation of transfer rather than repeated threshold adjustment.

## 2 Strict leave-one-domain-out baseline

We trained on two datasets and evaluated the third, with disjoint source-training, selection and calibration partitions. Each fold began from the original general-purpose foundation; an all-domain PAR checkpoint was never used to initialize a held-out fold. Source frequencies, sampling, checkpoint choice and calibration excluded the target. Models and thresholds were fixed before target evaluation. Six epochs per fold provided a matched recipe, although unequal source sizes mean different numbers of updates.

Calibrated scores were PA100K **0.797104**, PETA **0.777228** and Market1501 **0.817448**, macro **0.797260**, worst **0.777228**. Source-only calibration improved every target but left generalization gaps. Age, accessories and lower colors remained difficult. All source attributes had positives and negatives, but PA-held lower-orange had only four positives: sparse support matters without being a complete explanation of transfer failure.

A slow mounted-image copy design was replaced by verified private archives extracted once and reused across folds. This engineering change enabled the experiments without changing image bytes or partitions. Dataset images, features, private prediction arrays and weights are not included in Git.

## 3 Equal-domain sampling

We tested whether PA100K dominated source learning by changing only per-image weights from N_domain^-0.5 to N_domain^-1. This changes total domain probabilities from proportional to sqrt(N_domain) to 50/50. Architecture, initialization, loss/class frequencies, schedule and seed were held fixed; draw logs confirmed the changed exposure.

Calibrated macro fell to **0.796089** and worst to **0.776778**. PA gained 0.000460, PETA lost 0.000451 and Market lost 0.003522. PETA gained some mA but lost instance-F1, while Market lost both. The +0.003 macro/worst adoption margins failed. We retained square-root sampling. Reducing large-source image diversity while repeating small-source images is a plausible explanation, not a proven cause.

## 4 Frozen DINOv3 and complementary blending

We tested self-supervised DINOv3 ViT-B/16 because its visual appearance features might complement SigLIP2's language-aligned features. Direct access was unavailable; a maintainer-supported timm checkpoint passed provenance/license checks. An official DINOv2 fallback was prepared but cancelled when DINOv3 became available, so no completed DINOv2 comparative score is reported.

At a 320x160 canvas, DINOv3 produces CLS, four registers and 200 spatial patches. Six descriptors combine CLS, spatial mean, overlapping head/upper/lower means and spatial maximum. Attribute-specific attention selects the relevant regional evidence without a pose model. We froze the backbone, cached compact FP16 descriptors and trained source-only heads for up to 30 epochs with AdamW 3e-4, dropout 0.2 and batch 128. Selected fold epochs were 30/27/25.

Standalone calibrated DINO scored macro **0.754043**, worst **0.724861**, below SigLIP2. However, error overlap was incomplete: probability-residual correlations were approximately 0.747-0.776. We predeclared a **75% SigLIP2 / 25% DINO raw-probability blend**, followed by a new source-only 40-threshold vector, without searching weights on held-out targets.

Candidate A improved all three folds to **0.802375 / 0.780285 / 0.820231**, macro **0.800964**, worst **0.780285**. Macro mA rose modestly from 0.785674 to 0.786367; instance-F1 rose from 0.809226 to 0.816127. Most benefit was instance-level precision/recall rather than broad balanced-accuracy recovery. Age/accessories generally improved while clothing colors were mixed. A missed its primary macro threshold **0.8023**; we retained that negative gate but pursued one conservative adaptation experiment.

## 5 Candidate B partial adaptation

Each LODO fold started from the original DINOv3 foundation and its own strictly source-trained frozen head, never an all-domain head. Only blocks 10/11, final normalization and the head were trainable. We used AdamW backbone/head rates **5e-6 / 5e-5**, masked balanced BCE, original sampling, seed 2027, 320x160 preprocessing, flip p=0.5 and six epochs. The original SigLIP fold branch and 75/25 mixing stayed fixed.

Standalone DINO improved to macro **0.783481**, worst **0.752457**, still below SigLIP2. The ensemble scored **0.803448 / 0.781356 / 0.822720**, macro **0.802508**, worst **0.781356**, with macro mA **0.788772**. Every domain improved over A. The +0.001544 macro gain over A missed the predeclared **+0.002** margin; we did not substitute the older absolute threshold to redefine success. The consistent gains nevertheless motivated a separately authorized exploratory final all-domain comparison.

## 6 Candidate C robustness augmentation

This independent branch kept the original SigLIP2 architecture, foundation, sampler, loss and selection method. Only augmentation changed: bounded down/up sampling, mild gamma/brightness, JPEG compression and Gaussian blur, alongside the existing training augmentation. The intuition was to approximate surveillance image degradation while preserving geometry and colors.

All three six-epoch folds completed. Calibrated scores were **0.797197 / 0.776418 / 0.820613**, macro **0.798076**, worst **0.776418**. A small macro gain (+0.000816) came with a PETA regression (-0.000810), failing both +0.003 advancement margins. We rejected C and did not train an all-domain C model. More plausible augmentation is not automatically better for rare or fine-grained visual attributes.

## 7 Final models and official submission outcome

The exploratory all-domain A head was selected at epoch 30. Final B initialized the original pretrained backbone and A's all-domain head, using 97,669 training, 16,744 selection and 16,661 calibration rows. Only the final two blocks, normalization and head trained. Source improvements at epochs 6 and 7 justified the two declared extensions; all eight completed and epoch 8 was selected. Raw DINO source scores at epochs 6/7/8 were **0.838134 / 0.839221 / 0.840005**. Further extension was discussed but not run.

Final source-selection ensemble scores were **A 0.867806** and **B 0.869305**. These are development scores, not LODO or official results. New calibration used only source-calibration labels, and the original production SigLIP2 checkpoint/calibration remained unchanged. Independent reference checks reproduced DINO preprocessing/spatial tokens and the implemented blend, connecting the evaluated method to the model code.

Only the original SigLIP2 entry has a confirmed official score. The attempted A follow-up upload reset and reconnection failed during the reported site outage; further planned official evaluation was not completed. A/B improved every local held-out domain, and B gained **0.005248 macro** and **0.004127 worst-domain** over baseline. We consequently expected better generalization, but the exact score of an unscored official entry is unknown. Local gains cannot simply be added to 0.770450.

## 8 Limitations and improvements

Previously inspected LODO targets are development data. One seed, unequal source sizes/updates, sparse attributes and incomplete convergence limit causal/statistical conclusions. Exact organizer scorer edge cases remain unresolved, so primary local instance-F1 and mean-image-F1/unsupported-label sensitivities were considered. The published table clearly separates official, LODO and source-selection values.

Next work should repeat controlled comparisons across seeds, match training compute, improve rare-attribute coverage and examine calibration transfer. B's still-improving epoch-8 curve supports investigating limited source-selected continuation, while retaining a frozen reference and monitoring worst-domain mA. The strongest conclusion is useful complementary representation learning, not a guaranteed official score or a universal improvement from every training change.
