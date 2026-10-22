# Pedestrian attribute recognition under domain shift

UPAR 2027 Track 1 · Methodology, controlled experiments and final decisions · 22 October 2026

## 1. From recognition to transfer

We selected UPAR 2027 Track 1: predicting 40 binary pedestrian attributes from person crops, including age, gender, hair, clothing and accessories. Our central question became whether a strong classifier on familiar datasets could generalize to unfamiliar surveillance imagery. The project progressed from a semantic SigLIP2 baseline to controlled sampling, complementary DINOv3 representations, conservative adaptation and a robustness ablation.

Only the original model received a confirmed official evaluation: submission 1003877 scored 0.770450 Challenge_Avg. The later ensembles improved every local held-out domain, but their planned official evaluation was disrupted by the competition-site outage. We report their measured local results and expected transfer benefit without assigning an unobserved official score.

### Data integrity before model improvement

| Partition | Rows | Role |
| --- | --- | --- |
| Official training | 97,669 | PA100K 79,001; PETA 8,668; Market1501 10,000 |
| Official validation | 33,407 | Permitted development labels |
| Final source selection | 16,744 | Checkpoint selection |
| Final source calibration | 16,661 | Threshold fitting; disjoint from selection |

We preserved the exact 40-attribute order and official PETA filename mapping. Unknown labels were masked rather than converted to negatives. A pixel-duplicate audit found one group linking a training image to two validation rows; those two validation rows were excluded from later selection and calibration. Original data remained read-only. Exhaustive identity or near-duplicate separation was not established.

### Why we introduced leave-one-domain-out evaluation

The original known-domain validation score was 0.856185 at threshold 0.5, whereas the official score was 0.770450. This discrepancy motivated a test of transfer, rather than further optimization on familiar validation data. In each of three LODO folds, we trained on two datasets and evaluated the third. Source-training, source-selection and source-calibration partitions were disjoint.

Each fold used fresh general-purpose foundation weights, source-only label frequencies, sampling, checkpoint selection and calibration. We locked the checkpoint and thresholds before target evaluation. The six-epoch reference folds had unequal training sizes: 18,668 images with PA100K held out, 89,001 with PETA held out and 87,669 with Market1501 held out. Equal epochs therefore did not imply equal updates or convergence.

Later comparisons used already inspected LODO targets as development evidence. They are controlled comparisons, not untouched tests or statistical certification. Throughout the report, official, LODO and source-selection scores are kept separate.

## 2. Two complementary visual representations

### SigLIP2: semantic queries over pedestrian patches

The baseline uses general-purpose SigLIP2-Base-NaFlex. Aspect-aware preprocessing produces at most 256 valid 16×16 patches. Forty attribute queries are initialized from pretrained text embeddings, so labels such as clothing color and accessories begin with semantic information. The text encoder is used for initialization, not as an additional prediction branch.

An eight-head cross-attention layer lets each query attend to valid visual patches. Query residuals, a feed-forward network and normalization form an attribute-specific representation. Forty independent linear classifiers produce local logits; a masked spatial-mean linear branch adds global evidence. Padded patches are excluded from both attention and pooling.

<div align="center">z<sub>a</sub> = w<sub>a</sub><sup>T</sup>h<sub>a</sub> + b<sub>a</sub> + [W<sub>g</sub> mean<sub>valid</sub>(X)]<sub>a</sub></div>

X denotes visual patch features; hₐ is the residual cross-attention representation for attribute a. Dropout is applied during training.

### DINOv3: appearance features with regional attention

We chose self-supervised DINOv3 ViT-B/16 to complement SigLIP2’s language-aligned features. Direct checkpoint access was unavailable; a maintainer-supported public timm release of the original pretrained model passed provenance and license checks. An official DINOv2 fallback was prepared, then cancelled when DINOv3 became accessible; no completed DINOv2 comparison is claimed.

| Component | DINOv3 implementation |
| --- | --- |
| Input | Aspect-preserving 320×160 canvas; ImageNet normalization |
| Token layout | 1 CLS + 4 register tokens + 200 spatial patches (20×10) |
| Six descriptors | CLS; spatial mean; head mean; upper-body mean; lower-body mean; spatial maximum |
| Overlapping regions | Head 0–30%; upper 15–65%; lower 50–100% of resized person height |
| Padding and registers | Means weighted by patch/content overlap; registers never spatially pooled |
| Attribute head | LayerNorm → learned attention over six descriptors → 40 independent logits |

<div align="center">α<sub>ak</sub> = softmax<sub>k</sub>(q<sub>a</sub><sup>T</sup>d<sub>k</sub> / √768)     ;     z<sub>a</sub> = w<sub>a</sub><sup>T</sup>(Σ<sub>k=1…6</sub> α<sub>ak</sub>d<sub>k</sub>) + b<sub>a</sub></div>

dₖ is a normalized descriptor. Each attribute learns which regions matter; overlapping bands avoid requiring a pose estimator. The classifier uses dropout 0.2 during training.

Candidate A freezes the entire DINO backbone and trains only this head on cached FP16 descriptors. Candidate B starts from the original foundation and the appropriate frozen head, then updates only Transformer blocks 10/11, final normalization and the head. In LODO, every head is strictly source-trained; the all-domain head is used only for final all-domain training.

## 3. Shared learning and scoring rules

### Masked, class-balanced binary cross-entropy

For image i and attribute a, yᵢₐ is a binary label, vᵢₐ marks a finite valid 0/1 label, and zᵢₐ is the logit. Prevalence pₐ is computed from valid source-training labels only, before sampling. Positive and negative weights are exponential functions of that prevalence.

<div align="center">p<sub>a</sub> = Σ<sub>i</sub> v<sub>ia</sub>y<sub>ia</sub> / Σ<sub>i</sub> v<sub>ia</sub>     ;     w<sub>a</sub><sup>+</sup> = exp(1 − p<sub>a</sub>)     ;     w<sub>a</sub><sup>−</sup> = exp(p<sub>a</sub>)</div>

Rare positives receive a larger positive weight. Sampling probabilities do not redefine these class frequencies.

<div align="center">L = (1 / max(1,V)) Σ<sub>i,a</sub> v<sub>ia</sub>[w<sub>a</sub><sup>+</sup>y<sub>ia</sub> softplus(−z<sub>ia</sub>) + w<sub>a</sub><sup>−</sup>(1−y<sub>ia</sub>) softplus(z<sub>ia</sub>)]</div>

V = Σᵢ,ₐ vᵢₐ and softplus(t) = log(1 + exp(t)). The denominator counts valid labels, not weighted labels. Invalid targets are sanitized before the elementwise BCE is evaluated.

### Domain sampling and probability blending

<div align="center">weight(image i in domain d) proportional to N<sub>d</sub><sup>α−1</sup>     therefore     P(domain d) proportional to N<sub>d</sub><sup>α</sup></div>

The original α = 0.5 yields square-root-domain sampling; the ablation α = 0 gives equal probability to each source domain. Images are sampled with replacement.

<div align="center">p<sub>ensemble,a</sub> = 0.75 σ(z<sub>SigLIP,a</sub>) + 0.25 σ(z<sub>DINO,a</sub>)     ;     ŷ<sub>a</sub> = 1[p<sub>ensemble,a</sub> &gt; t<sub>a</sub>]</div>

The fixed mixture combines raw sigmoid probabilities. It does not blend binary predictions or independently calibrated outputs. Forty global thresholds are fitted on source-calibration data only.

| Recipe | Trainable parameters / schedule | Learning rates and batch |
| --- | --- | --- |
| Original SigLIP2 | Head warmup; upper 4 blocks + norm; 12 epochs; EMA 0.9995; best epoch 7 | AdamW: backbone 1e−5, head 1e−4; effective batch 64 |
| SigLIP LODO / C | Same architecture and loss; 6 epochs per fold; C changes augmentation only | Original optimizer and sampler; seed 2027 |
| Frozen DINO / A | Head only; max 30 epochs; patience 5; LODO best 30/27/25 | AdamW head 3e−4; batch 128; dropout 0.2 |
| Adapted DINO / B | Blocks 10/11 + norm + head; 6 LODO epochs; no EMA introduced | AdamW backbone 5e−6, head 5e−5; batch 16×4 = 64 |

### What the reported score means

<div align="center">mA = mean<sub>a</sub>[(TPR<sub>a</sub> + TNR<sub>a</sub>)/2]     ;     F1 = 2PR/(P+R)     ;     C<sub>d</sub> = 2 mA<sub>d</sub> F1<sub>d</sub> / (mA<sub>d</sub> + F1<sub>d</sub>)</div>

P and R are mean-image precision and recall over valid attributes. Primary F1 is their harmonic mean, not mean per-image F1. Local mA excludes attributes without both positive and negative support. Macro = mean of the three domain scores C_d; worst = min C_d.

Calibration uses two coordinate passes over 11 fixed thresholds from 0.22 to 0.78, with a 0.005 mean absolute deviation penalty from 0.5. The exact organizer’s empty-label conventions are private; alternative F1 and unsupported-label sensitivities were considered. Local values are a faithful available reproduction, not a claim of complete scorer equivalence.

## 4. What the controlled comparisons measured

| Calibrated method | PA100K | PETA | Market | Macro | Worst |
| --- | --- | --- | --- | --- | --- |
| SigLIP2 baseline | 0.797104 | 0.777228 | 0.817448 | 0.797260 | 0.777228 |
| Equal-domain | 0.797565 | 0.776778 | 0.813926 | 0.796089 | 0.776778 |
| Frozen DINO alone | 0.754206 | 0.724861 | 0.783061 | 0.754043 | 0.724861 |
| A: frozen ensemble | 0.802375 | 0.780285 | 0.820231 | 0.800964 | 0.780285 |
| B: adapted DINO alone | 0.780643 | 0.752457 | 0.817344 | 0.783481 | 0.752457 |
| B: adapted ensemble | 0.803448 | 0.781356 | 0.822720 | 0.802508 | 0.781356 |
| C: robustness | 0.797197 | 0.776418 | 0.820613 | 0.798076 | 0.776418 |

*All values are local Challenge_Avg. Each target was excluded from that fold’s training, selection and calibration. Six-decimal precision identifies the recorded result; it does not imply statistical certainty.*

### The baseline: calibration helps, but transfer remains uneven

The baseline’s raw LODO macro was 0.789605; source-only calibration increased it to 0.797260 and helped all three targets. PETA remained the worst domain at 0.777228. This supports calibration transfer as useful, but does not show that calibration removes visual domain shift. A disjoint known-source calibration audit scored 0.863482, close to the in-sample calibrated value 0.864952; that small optimism could not by itself explain the much larger official gap.

All 40 source attributes had both positive and negative examples in every fold. Sparse supervision was nevertheless important: PA-held lower-orange had only four source positives. Age, accessories and lower clothing colors remained difficult. We therefore avoided a blanket “missing source labels” explanation or attributing every error to the backbone.

### Equal-domain sampling: an intuitive change that did not help

We hypothesized that the large PA100K source could dominate learning. Changing only α from 0.5 to 0 made the two source domains equally likely, while preserving class-frequency computation and BCE weights. Actual sampling draws were recorded for the new run; baseline exposure probabilities are expected values because its actual draws were not logged.

Calibrated macro fell by 0.001171 (0.117 percentage points), and worst-domain performance fell by 0.000451. PA100K improved slightly, but PETA regressed and Market1501 fell by 0.003522. The raw macro also fell, from 0.789605 to 0.786694. Equal weighting therefore did not reveal a consistent gain hidden by calibration.

The predeclared +0.003 macro and worst-domain adoption margins were not met. We retained square-root sampling. Repeating a smaller source more often while reducing exposure to a larger source’s diversity is a plausible explanation; this single-seed experiment does not establish that causal mechanism.

### Reading the table without conflating experiments

The DINO standalone rows test its strength as a replacement; the ensemble rows test its usefulness as a complement. Those questions have different answers. Likewise, a small macro gain with a worse weakest domain is not equivalent to broad improvement. The following analysis uses the same folds and scoring conventions rather than comparing unrelated source-validation scores.

## 5. Complementarity helped more than replacement

| Calibrated macro | mA | Instance-F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| SigLIP2 baseline | 0.785674 | 0.809226 | 0.787004 | 0.833028 |
| A: frozen ensemble | 0.786367 | 0.816127 | 0.794574 | 0.839026 |
| B: adapted ensemble | 0.788772 | 0.816744 | 0.794937 | 0.839878 |

### Candidate A: weaker alone, useful in a fixed ensemble

Frozen DINO alone scored only 0.754043 macro. Its errors were partly distinct, however: probability-residual correlations with SigLIP were approximately 0.747–0.776. The predeclared 75/25 blend improved all three domains. Its macro gain over SigLIP was 0.003704 (0.370 percentage points), and its worst-domain gain was 0.003057.

The gain was driven more by instance-F1 (+0.006901) than by mA (+0.000693). Both precision and recall improved. PETA’s balanced accuracy declined despite its Challenge_Avg gain, so this was not a broad recovery of every attribute. Age and accessories generally benefited, while clothing-color changes were mixed. Incomplete error correlation supports complementarity; it does not guarantee benefit for every group.

A’s macro 0.800964 missed the predeclared 0.8023 promotion threshold, although its worst domain exceeded 0.7772 and no domain regressed. We preserved that failed gate. The near-useful complementary signal motivated one controlled partial-adaptation experiment, rather than an ensemble-weight search.

### Candidate B: conservative adaptation improved all three domains

We unfroze only the final two DINO blocks, final normalization and head. The corresponding source-trained frozen head initialized each fold, with original pretrained weights elsewhere. Every fold completed six epochs; source selection chose epochs 6/5/5. Standalone macro rose to 0.783481, still below SigLIP, so the evidence continued to favor an ensemble rather than replacement.

The B ensemble improved over A by 0.001073 on PA100K, 0.001070 on PETA and 0.002489 on Market1501. Macro reached 0.802508 and worst-domain performance 0.781356. Compared with A, macro mA increased by 0.002405 and instance-F1 by 0.000617. Thus adaptation added a more balanced-accuracy-oriented improvement than frozen blending alone.

B’s +0.001544 macro gain over A remained below its predeclared +0.002 advancement margin. It would be incorrect to substitute A’s older absolute threshold and call B a successful confirmatory experiment. We recorded the negative research decision; the later all-domain B run was separately authorized as an exploratory competition candidate.

### What we selected

We kept the original SigLIP2 checkpoint as the scored fallback, retained the fixed 75/25 mixture for A and B, and selected B as the strongest local transfer candidate. No hidden-test scores or target-based threshold/mixture searches were used to select these settings. The consistent domain gains are encouraging development evidence, constrained by one seed and repeated target inspection.

## 6. Final candidates, setbacks and lessons

### Candidate C: robustness augmentation was rejected

The independent SigLIP branch changed only augmentation. Its bounded policy used down/up sampling (p=0.30, scale 0.65–1), JPEG (p=0.30, quality 60–95), blur (p=0.15, σ 0.2–0.7), gamma (p=0.20, 0.9–1.1) and brightness (p=0.50, 0.9–1.1). The intention was to approximate surveillance degradation without changing geometry or strongly distorting colors.

C’s macro 0.798076 was only 0.000816 above baseline; PETA, the worst domain, fell to 0.776418. Both +0.003 advancement margins failed. We rejected C and did not train a final all-domain version. Even mild degradation can remove small accessory or fine-color evidence; that remains a hypothesis, not a demonstrated explanation of the regression.

### Final all-domain A and B: source-selected exploratory models

A trained its frozen head on all 97,669 training images and selected epoch 30. Final B initialized the original DINO foundation plus A’s all-domain head, then adapted the same limited components as in LODO. It reused the disjoint 16,744 selection / 16,661 calibration partitions. New ensemble calibration used calibration labels only; the original production SigLIP weights and thresholds stayed separate and unchanged.

| Final B epoch | Raw DINO source-selection score | Change from preceding row |
| --- | --- | --- |
| 6 | 0.838134 | — |
| 7 | 0.839221 | +0.001087 |
| 8 (selected) | 0.840005 | +0.000783 |

Epochs 6 and 7 each exceeded the previous best by at least 0.0001, satisfying the declared extension rule. Eight epochs completed and epoch 8 was selected. The still-positive but diminishing source gains justify investigating limited continuation in future work, not assuming that extra epochs would improve unseen-domain performance. No further extension was run.

| Evaluation context | A ensemble | B ensemble | Interpretation |
| --- | --- | --- | --- |
| Final source selection | 0.867806 | 0.869305 | Known-source development; not LODO |
| Calibrated LODO macro | 0.800964 | 0.802508 | Transfer comparison on three held-out domains |
| Official evaluation | Not confirmed | Not confirmed | Only original SigLIP2: 0.770450 |

The completed original submission 1003877 reported mA 0.773647, instance-F1 0.767279, precision 0.741799 and recall 0.807915, alongside Challenge_Avg 0.770450. These are official measurements; the A/B numbers above are local measurements from different evaluation contexts.

The A follow-up upload reset and reconnection failed during the reported competition-site outage. Further planned official evaluation could not be completed. A and B improved every local held-out domain; B gained 0.005248 macro and 0.004127 worst-domain over baseline. This supports an expectation of better generalization if evaluated officially. It does not determine an official score, and those local gains cannot simply be added to 0.770450.

### Challenges, reproducibility and the next useful improvement

A slow individual-file data workflow was replaced with verified private archives extracted once. Checkpoint access, register/spatial indexing, preprocessing parity, missing-label masking and duplicate exclusions required careful verification. Independent checks tied the native models and raw-probability blend to their reference implementations; the clean code tree passed 71 CPU tests. The repository contains working code, compact settings and aggregate results, while private images, features and weights remain excluded.

The strongest lesson is that complementary representations helped more reliably than equalizing domains or adding generic degradation. Next work should repeat the fixed comparisons across seeds, match training updates across unequal source sizes, improve rare-attribute coverage and test source-only calibration transfer. Limited B continuation is reasonable to investigate while monitoring mA and worst-domain behavior; a universally better model or a higher unobserved official score is not established.

Code and measured aggregates: assignment/RESULTS.csv; configs/experiments.json; assignment/EXPERIMENTS_AND_DECISIONS.md. Upstream task: github.com/speckean/UPAR-Challenge-2027. Foundations: google/siglip2-base-patch16-naflex; github.com/facebookresearch/dinov3; timm/vit_base_patch16_dinov3.lvd1689m. DINOv3 route: facebookresearch/dinov3 issue 375. Model notices are retained in the repository.
