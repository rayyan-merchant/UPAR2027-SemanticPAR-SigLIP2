# UPAR 2027 pedestrian attribute recognition

**Assignment code and results** · Track 1 · SigLIP2 and complementary DINOv3

We predict 40 pedestrian attributes across PA100K, PETA and Market1501, and test transfer to datasets excluded from training. The project starts with a semantic SigLIP2 classifier, investigates sampling and domain shift, then evaluates complementary DINOv3 features and a controlled robustness policy.

**Official submission:** 1003877, Challenge_Avg **0.770450**. The later ensembles improve local held-out performance, but have no confirmed official scores after the interrupted follow-up evaluation during the reported site outage.

## Assignment deliverables

| Required item | Open here |
|---|---|
| Brief task description and approach | [Task and approach](assignment/TASK_AND_APPROACH.md) |
| Short methodology and evaluation report | [Methodology and results PDF](assignment/UPAR2027_Assignment_Report.pdf) · [Text version](assignment/UPAR2027_Assignment_Report.md) |
| Working code | [SigLIP2](upar/) · [DINOv3](dino/) · [Tests](tests/) |
| Experimental results | [Results table](assignment/RESULTS.csv) · [Experiments and decisions](assignment/EXPERIMENTS_AND_DECISIONS.md) |
| Setup and reproduction | [Reproduction guide](docs/REPRODUCING.md) · [Reviewer guide](docs/REVIEWER_GUIDE.md) |

This private GitHub repository is the code hand-in. The submitter will grant the evaluator access. Internal lab notes, execution logs, cloud receipts and analysis dumps are intentionally excluded from the submitted tree.

## Method

The original model uses **SigLIP2-Base-NaFlex** with text-initialized attribute queries, masked patch attention and a global visual residual. Training uses masked class-balanced BCE, square-root-domain sampling and source-only selection/calibration. Unknown labels are masked, not converted to negatives.

The complementary **DINOv3 ViT-B/16** branch pools CLS, spatial mean, overlapping head/upper/lower means and spatial maximum into six descriptors. Attribute-specific attention predicts 40 logits. We compare a frozen backbone with conservative final-two-block adaptation. The fixed ensemble blends **75% raw SigLIP2 probabilities and 25% raw DINO probabilities** before fitting one source-only threshold vector.

Every LODO fold excludes its target dataset from training, label frequencies, checkpoint selection and calibration. Later comparisons are development experiments on previously inspected targets, not untouched certification. Key experiment settings are in [configs/experiments.json](configs/experiments.json).

## Results

### Official evaluation

| Model | Submission | Challenge_Avg | mA | Instance-F1 |
|---|---:|---:|---:|---:|
| Original SigLIP2 | 1003877 | **0.770450** | 0.773647 | 0.767279 |

### Local calibrated LODO evaluation

| Method | PA100K | PETA | Market1501 | Macro | Worst |
|---|---:|---:|---:|---:|---:|
| SigLIP2 baseline | 0.797104 | 0.777228 | 0.817448 | 0.797260 | 0.777228 |
| Equal-domain sampling | 0.797565 | 0.776778 | 0.813926 | 0.796089 | 0.776778 |
| Frozen DINOv3 standalone | 0.754206 | 0.724861 | 0.783061 | 0.754043 | 0.724861 |
| A frozen ensemble | 0.802375 | 0.780285 | 0.820231 | 0.800964 | 0.780285 |
| B adapted ensemble | 0.803448 | 0.781356 | 0.822720 | **0.802508** | **0.781356** |
| C robustness augmentation | 0.797197 | 0.776418 | 0.820613 | 0.798076 | 0.776418 |

We retained square-root sampling and rejected the robustness policy's worst-domain regression. The ensembles improved every held-out domain; B gained 0.005248 macro over baseline. Both original conservative advancement margins remained unmet, and final A/B training was a separate exploratory decision. The stronger local results support an expectation of improved generalization, not an unmeasured official score. [Full interpretation](assignment/EXPERIMENTS_AND_DECISIONS.md)

## Quick start

Use Python 3.12 to match the recorded local code checks.

```bash
git clone https://github.com/rayyan-merchant/UPAR2027-SemanticPAR-SigLIP2.git
cd UPAR2027-SemanticPAR-SigLIP2
python scripts/verify_assignment_results.py
```

The verifier uses only Python's standard library. It checks the **73 published aggregate rows**, their macro/worst arithmetic, the official entry and the stated comparison decisions. It does not reproduce raw image inference or independently re-score private predictions.

For the CPU correctness suite:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
python -m pip install -r requirements-dino.txt
python -m pytest -q tests
```

The full suite checks model shapes, valid-label masking, attribute order, sampling, fold isolation, calibration and offline API behavior. Training and image-level evaluation require separately authorized datasets and matching model assets; see [REPRODUCING.md](docs/REPRODUCING.md).

## Code layout

```text
assignment/           Report, task description and compact research results
configs/              Model and experiment settings
upar/                 SigLIP2 model, preprocessing, training, metrics and calibration
dino/                 DINOv3, regional head, frozen/partial training and ensemble APIs
kaggle/               Archive staging and fold execution utilities
scripts/              Fold preparation, metric helpers and assignment verification
tests/                CPU correctness and isolation tests
submission_template/  Original inference API and metadata contract
docs/                 Setup and reviewer instructions
```

## Data and licensing

The official collection has 97,669 training and 33,407 validation rows. The known exact-pixel validation overlaps are excluded from later source-selection/calibration. Images, private annotations/identities, trained weights, feature caches, prediction arrays and credentials are not included. Private research logs and receipts are not part of the assignment repository.

Use the [official UPAR data](https://github.com/speckean/UPAR-Challenge-2027), [Google SigLIP2 foundation](https://huggingface.co/google/siglip2-base-patch16-naflex) and [maintainer-supported timm DINOv3 foundation](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m) under their original terms. [Third-party notices](THIRD_PARTY_NOTICES.md) and [model licenses](dino/licenses/) are retained. No cloud training or competition upload is started by the review commands above.
