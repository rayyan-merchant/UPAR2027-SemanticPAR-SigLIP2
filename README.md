# UPAR 2027 pedestrian attribute recognition

This assignment implements and evaluates 40-attribute pedestrian recognition with a semantic SigLIP2 model, strict leave-one-domain-out diagnostics and complementary DINOv3 regional attention. **The only confirmed official submission is 1003877: Challenge_Avg 0.770450.** Later candidates are locally evaluated or packaged, not officially scored. The failed Candidate A upload and Candidate B runtime gate are documented, not hidden.

## Assignment deliverables

1. **Task and approach:** [brief description](assignment/TASK_AND_APPROACH.md) and the opening of the [three-page report](assignment/UPAR2027_Assignment_Report.pdf); [editable report](assignment/UPAR2027_Assignment_Report.md).
2. **Working code and experimental results:** `upar/`, `dino/`, `kaggle/`, `scripts/`, `tests/`, [RESULTS.csv](assignment/RESULTS.csv) and retained aggregate `reports/` evidence.
3. **Methodology, evaluation, challenges and improvements:** the report, plus the [complete experiment and decision record](assignment/EXPERIMENTS_AND_DECISIONS.md) for the details that cannot fit into three pages.

## Results and final selection

| Experiment | Evaluation | Challenge_Avg | Worst domain | Decision |
|---|---|---:|---:|---|
| Original SigLIP2 | Official submission 1003877 | **0.770450** | Not comparable to LODO | Official evaluated fallback |
| Original SigLIP2 | Three-fold calibrated LODO | 0.797260 | 0.777228 | Locked baseline |
| Equal-domain sampling | Same LODO protocol | 0.796089 | 0.776778 | Reject; retain square-root sampler |
| Frozen DINOv3 standalone | Same LODO partitions | 0.754043 | 0.724861 | Reject standalone |
| Candidate A frozen 75/25 ensemble | Same LODO partitions | 0.800964 | 0.780285 | Useful; missed 0.8023 gate; exploratory package |
| Candidate B adapted 75/25 ensemble | Same LODO partitions | 0.802508 | 0.781356 | Best local transfer; missed +0.002 advance margin; separately authorized final run |
| Candidate C augmentation | Same LODO protocol | 0.798076 | 0.776418 | Reject worst-domain regression |

No local score is a hypothetical official result. B completed eight all-domain epochs, selected epoch 8 and passed source/checkpoint/reference checks. Its T4 complete-workload probe timed out at 300 seconds after last saved progress 26,624/29,248 images; actual A10 timing remains unknown. A's upload produced no confirmed ID during the reported site outage. Original 1003877 and its assets remain preserved.

## Repository map

| Directory | Purpose |
|---|---|
| `assignment/` | Three-page PDF, editable narrative, full decision record and reproducible result table |
| `upar/` | SigLIP2 architecture, masked loss, sampling, metrics, calibration, training and offline submission API |
| `dino/` | Native DINOv3, six-descriptor regional head, frozen/limited training and separate A/B inference wrappers |
| `kaggle/` | Archive staging, watchdogs, source-only gates and independent fold scheduling |
| `configs/` | Recorded recipe; executed configurations remain in reports |
| `scripts/` | Preparation, audits, packaging, result verification and independent metric reproduction |
| `tests/` | CPU correctness and isolation/contract checks; no cloud training is started by pytest |
| `reports/` | Aggregate metrics, controls, hashes and receipts; no image pixels or private prediction arrays |
| `docs/` | Evidence index and reproduction guide; root research reports retain historical decisions |
| `submission_template/` | Original required inference metadata and starter contract |

## Quick start without datasets or GPUs

Use Python 3.12 to match the recorded local checks; deployment was also checked separately on Python 3.11. The tested local environment used torch 2.4.1 CPU, Transformers 4.57.1 and timm 1.0.24. Kaggle cloud jobs recorded their actual environments separately. The official runtime is Python 3.11 / torch 2.4.1; a current Kaggle container is not an equivalent timing environment.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
python -m pip install -r requirements-dino.txt
python -m pytest -q tests
python scripts/verify_assignment_results.py
```

The last command is standard-library-only: it reconstructs the assignment result table from checked-in aggregate evidence and checks the stated decisions. It does **not** re-evaluate private predictions or train models. Full raw-prediction audits were executed previously; their receipts and scripts are retained. See [reproduction instructions](docs/REPRODUCING.md) for those data/weight-dependent steps and the [evidence index](docs/EXPERIMENT_INDEX.md) for each claim.

## Data and weight access

Training requires authorized official UPAR data and verified general-purpose foundation weights. Obtain data through the [official challenge repository](https://github.com/speckean/UPAR-Challenge-2027), retaining each original dataset's terms. Restricted images, annotations containing sample identities, cached features, private predictions, trained weights and credentials are intentionally not distributed in this source repository or assignment ZIP. Model and submission SHA256 values are retained in the reports. The approximately 651 MB A/B competition ZIPs remain separate private artifacts, not part of the assignment code ZIP.

Do not automatically run cloud builders from this README: they allocate paid/quota-limited compute when pushed. Do not use an all-domain PAR checkpoint to initialize a LODO fold. Do not fit calibration on a target domain. Do not tune weights or thresholds with hidden scores.

## Deployment and licensing

Inference uses native torch, NumPy and Pillow with no downloads, Transformers, timm, safetensors or text encoder at runtime. A/B blend 75% raw SigLIP probabilities with 25% raw DINO probabilities, then apply a separate source-fitted calibration. Required 40-attribute order and original metadata are preserved. DINOv3 Meta and timm notices are in `dino/licenses/`; source data retains its own restrictions. See [third-party notices](THIRD_PARTY_NOTICES.md). Do not interpret repository visibility as permission to redistribute dataset images or model weights.
