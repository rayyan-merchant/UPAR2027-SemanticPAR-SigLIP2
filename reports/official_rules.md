# Overview

# <p style="text-align: center;">Real-World Surveillance: Applications and Challenges, 7th <br>@ WACV2027<br>UPAR Challenge Track 1: Pedestrian Attribute Recognition</p>


## News
- Oct 8, 2026: Update from CPU to GPU runners:  Nvidia A10 GPU, 24 GB VRAM, 20 CPU Cores, 40 GB RAM, time limit per run: 5 minutes
- Oct 6, 2026: We have clarified the rules. Please review them to ensure compliance.
- Oct 2, 2026: New code & fact sheet submission deadline: **Oct 26, 2026, AoE**. Due to strict WACV deadlines, we must move up the deadline.
- Sep 30, 2026: [Track3](https://www.codabench.org/competitions/18273/) on Human Pose Estimation is now live


## Challenge & Workshop
Track 1 of the **UPAR Challenge** at the Real-World Surveillance workshop
(RWS), WACV 2027.

Everything about this track lives on this Codabench page: task, data,
evaluation, submission format, schedule and rules. Announcements go out by
email to registered participants.

See also the workshop website and Tasks 2 and 3 of the UPAR Challenge.
The RWS workshop welcomes both regular submissions on the topic of real-world surveillance and submissions as part of the Challenge:
- [RWS Workshop](https://vap.aau.dk/rws/)
- [Challenge Task 1: Pedestrian Attribute Recognition](https://www.codabench.org/competitions/18180/)
- [Challenge Task 2: Attribute-Based Person Retrieval](https://www.codabench.org/competitions/18196/)
- [Challenge Task 3: Human Pose Estimation](https://www.codabench.org/competitions/18273/)


## The task

Pedestrian attribute recognition under domain shift. For every pedestrian crop
predict **40 binary attributes**: age, gender, hair length, upper and lower body
clothing length, colour and type, and accessories.

The challenge builds on the UPAR dataset, which unifies the annotations of
several public pedestrian datasets. The test images come from a domain your
model has never seen, so the method has to generalise beyond its training
data.

You submit **code, not predictions**. Your model runs inside the evaluation
container on a hidden test set, and only the resulting metrics are published.

## Getting started

1. Read the **Data** page and download the training data and example submission. [Starter Kit](https://github.com/speckean/UPAR-Challenge-2027)
2. Take the **example submission** as your template. It shows the
   functions the platform calls.
3. Put your model and weights into the submission folder, zip the **contents**
   of the folder so `run.py` is at the archive root, and upload it under
   **My Submissions**.

See the **Submission** page for the API your `run.py` must expose, and the
**Evaluation** page for the metrics.

Submissions run in `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
(Python 3.11, torch 2.4.1, torchvision, numpy, Pillow) with **no network
access**. Ship everything your model needs inside the zip.

## Code submission and winner selection
After the challenge deadline, participating teams are required to submit their code, models, and a short description of their method (fact sheet) by **Oct 26, 2026, AoE** to **upar.challenge@gmail.com**. A panel of experts will review the entries. The executable/source code should include trained models, training code, and necessary parameters to ensure reproducibility. Teams that do not comply are removed from the final ranking.

Note: Submitted codes will not be saved on the evaluation servers. Final submission codebases provided to the organizers will be used only for reproducibility checks and winner verification, and will not be shared publicly or with other participants at any stage.

## Paper submission
We strongly encourage all challenge participants to submit a technical paper to the RWS workshop.
Your paper should detail your methodology, experiments, and results on the challenge dataset.
Accepted papers will be presented at the workshop and published in the WACV workshop proceedings.
This is an excellent opportunity to share your insights with the community and gain visibility for your work.

Submission Guidelines:
- Format: [WACV format](https://wacv.thecvf.com/Conferences/2026/AuthorReviewerGuides)
- Submission System: [Openreview](https://openreview.net/group?id=thecvf.com/WACV/2027/Workshop/RWS)
- Challenge paper deadline: **Oct 23, 2026, AoE**

## Schedule

| Date | Event |
| --- | --- |
| Sep 21, 2026 | Challenge opens, training data available |
| Oct 21, 2026, AoE | Challenge submission deadline |
| Oct 23, 2026, AoE | Paper submission deadline |
| Oct 26, 2026, AoE | Code & fact sheet submission deadline, code verification starts |
| Jan 4/5, 2027 | RWS Workshop at WACV 2027 |

## Citation

```bibtex
@inproceedings{specker2023upar,
  title={UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval},
  author={Specker, Andreas and Cormier, Mickael and Beyerer, Jurgen},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2023}
}

@inproceedings{cormier2023upar,
  title={UPAR challenge: pedestrian attribute recognition and attribute-based person retrieval-dataset, design, and results},
  author={Cormier, Mickael and Specker, Andreas and Jacques, Julio CS and Florin, Lucas and Metzler, J{\"u}rgen and Moeslund, Thomas B and Nasrollahi, Kamal and Escalera, Sergio and Beyerer, J{\"u}rgen},
  booktitle={2023 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  pages={166--175},
  year={2023},
  organization={IEEE}
}

@inproceedings{cormier2024upar,
  title={Upar challenge 2024: Pedestrian attribute recognition and attribute-based person retrieval-dataset, design, and results},
  author={Cormier, Mickael and Specker, Andreas and Junior, Julio and CS, Jacques and Moritz, Lennart and Metzler, J{\"u}rgen and Moeslund, Thomas B and Nasrollahi, Kamal and Escalera, Sergio and Beyerer, J{\"u}rgen},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={359--367},
  year={2024}
}
```

# Data

# Data

## Training data

Training data and starter kit can be found here: [Starter Kit](https://github.com/speckean/UPAR-Challenge-2027)
The use of external data to fine-tune PAR is not permitted. Pre-trained models trained on common datasets such as ImageNet, or foundation models/VLMs that were not specifically trained for PAR, are permitted.

## Attributes

40 binary attributes, in this exact order. Your model returns one probability
per attribute in this order.

```text
 0 Age-Young                  20 LowerBody-Length-Short
 1 Age-Adult                  21 LowerBody-Color-Black
 2 Age-Old                    22 LowerBody-Color-Blue
 3 Gender-Female              23 LowerBody-Color-Brown
 4 Hair-Length-Short          24 LowerBody-Color-Green
 5 Hair-Length-Long           25 LowerBody-Color-Grey
 6 Hair-Length-Bald           26 LowerBody-Color-Orange
 7 UpperBody-Length-Short     27 LowerBody-Color-Pink
 8 UpperBody-Color-Black      28 LowerBody-Color-Purple
 9 UpperBody-Color-Blue       29 LowerBody-Color-Red
10 UpperBody-Color-Brown      30 LowerBody-Color-White
11 UpperBody-Color-Green      31 LowerBody-Color-Yellow
12 UpperBody-Color-Grey       32 LowerBody-Color-Other
13 UpperBody-Color-Orange     33 LowerBody-Type-Trousers&Shorts
14 UpperBody-Color-Pink       34 LowerBody-Type-Skirt&Dress
15 UpperBody-Color-Purple     35 Accessory-Backpack
16 UpperBody-Color-Red        36 Accessory-Bag
17 UpperBody-Color-White      37 Accessory-Glasses-Normal
18 UpperBody-Color-Yellow     38 Accessory-Glasses-Sun
19 UpperBody-Color-Other      39 Accessory-Hat
```

All 40 attributes are scored. Always return all 40 values.

## Test data

The test set is hidden. It is mounted into your container, read-only, with this
layout:

```text
input_data/
â”œâ”€â”€ images.csv      one image path per line, no labels
â”œâ”€â”€ images/         the image files, at the paths listed in images.csv
â”‚   â””â”€â”€ <dataset>/...
â””â”€â”€ task.json       attribute names, number of images
```

The images are pedestrian crops of varying size and aspect ratio.

You do not have to parse these files yourself: the ingestion program does it
and hands your code one ready-made `sample` dict per image. See the
**Submission** page.

# Evaluation

# Evaluation

The protocol and metrics are those of the previous UPAR challenge edition
(WACV 2024). The scoring program reproduces the original evaluation script
exactly.

## Decision threshold

Your model returns a probability in `[0, 1]` per attribute. An attribute counts
as **present if its probability is above 0.5**. Calibrate your outputs
accordingly. Values outside `[0, 1]` fail the submission.

## Metrics

| Leaderboard | Key | Meaning |
| --- | --- | --- |
| **Avg** | `Challenge_Avg` | harmonic mean of `mA` and `Instance_F1` - **the ranking metric** |
| mA | `mA` | label-based mean accuracy |
| Label_F1 | `Label_F1` | label-based F1 |
| Inst_Acc | `Instance_Acc` | instance-based accuracy |
| Inst_Prec | `Instance_Prec` | instance-based precision |
| Inst_Rec | `Instance_Rec` | instance-based recall |
| Inst_F1 | `Instance_F1` | instance-based F1 |

**Label-based** metrics are computed per attribute and then averaged over the
attributes, so rare attributes weigh as much as frequent ones:

```text
mA       = mean over attributes of ( TP / (TP + FN)  +  TN / (TN + FP) ) / 2
Label_F1 = mean over attributes of the attribute's F1
```

**Instance-based** metrics are computed per image, over the set of attributes
predicted present `P` and the set actually present `G`, and then averaged over
the images:

```text
Instance_Acc  = |P âˆ© G| / |P âˆª G|
Instance_Prec = |P âˆ© G| / |P|
Instance_Rec  = |P âˆ© G| / |G|
Instance_F1   = 2 Â· Prec Â· Rec / (Prec + Rec)
```

The ranking metric balances both views:

```text
Challenge_Avg = 2 Â· mA Â· Instance_F1 / (mA + Instance_F1)
```

## Datasets

If the test set contains images from several datasets, every dataset is
scored on its own, over all 40 attributes. Every leaderboard value is the
**mean over the datasets**, so a small dataset weighs as much as a large one.
The standard deviation over the datasets is reported in the detailed results.

Attributes whose ground truth is marked invalid in a dataset are excluded from
the evaluation of that dataset.

## Leaderboard

Your **latest** submission is the one shown on the leaderboard, not your best.

# Submission

# Submission

You upload a zip containing your model, the platform runs it against data you never see.

## The zip

Zip the **contents** of your submission folder, not the folder itself:

```bash
cd my_submission
zip -r ../my_submission.zip . -x "*/__pycache__/*" "*.DS_Store" ".git/*"
unzip -l ../my_submission.zip | head    # run.py must be at the root
```

Required at the root of the archive:

```text
run.py           your entry point
metadata.yaml    copy it unchanged from the example submission
<anything else>  weights, modules, configs
```

## The API

The ingestion program imports `run.py` and calls:

```python
def load_model() -> None:
    """Optional. Called once before inference. Load your weights here."""

def predict_image(sample: dict) -> list[float]:
    """Called once per image. Return 40 probabilities."""
```

`sample` is:

```python
{
  "index":           0,                        # row in images.csv
  "image":           "<dataset>/images/0.jpg",     # path as listed in images.csv
  "image_path":      "/app/input_data/images/<dataset>/images/0.jpg",   # absolute, readable
  "attribute_names": ["Age-Young", "Age-Adult", ...],   # the 40 names, in order
}
```

Return **one probability per attribute, in the order of `attribute_names`**: a
list, tuple, numpy array or tensor of 40 floats in `[0, 1]`, or a dict mapping
every attribute name to its probability. An attribute counts as present above
0.5 (see the Evaluation page). NaN, values outside `[0, 1]` or a wrong length
fail the submission with a message in the ingestion log.

### Batched inference

Per-image calls are simple but slow on a GPU. Define `predict_batch` instead
of, or in addition to, `predict_image` and it is used:

```python
BATCH_SIZE = 64    # optional, default 32

def predict_batch(samples: list[dict]) -> list[list[float]]:
    """One row of 40 probabilities per sample, in order."""
```

## Runtime environment

- **No network access.** Ship your weights inside the zip. Nothing is downloaded at run time.
- The image is `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`: Python 3.11,
  torch 2.4.1, torchvision, numpy, Pillow. Anything else your code imports must
  travel in the zip.
- There is a wall-clock limit for the whole run. The ingestion program logs
  its progress and per-image timing.
- The working directory is your submission folder, so relative paths to your
  weights work.

# Terms and Conditions

## 1. The challenge

The UPAR Challenge at the Real-World Surveillance workshop, WACV 2027, is a
skill-based contest with three independent tracks. These terms cover **Track 1:
Pedestrian Attribute Recognition**. The challenge is run entirely on Codabench:
this benchmark page is the official source for the task description, data,
evaluation, schedule, rules and announcements.

## 2. Eligibility

- Participation is open to everyone except the organizers and members of their
  research groups.
- You may participate alone or in a team. Each person may belong to one team
  only, and each team may use one Codabench account only.
- To be eligible for ranking and prizes you must register with your real
  identity and affiliation.

## 3. Training data, pre-training, and post-processing

- Models must be trained **only on the training data provided by the
  organizers**.
- The use of external data to fine-tune PAR is not permitted.
- Pre-trained models trained on common datasets such as ImageNet,
  or foundation models/VLMs that were not specifically trained for PAR, are permitted.
- **Any use of the test images is prohibited**, including training,
  pseudo-labelling, manual annotation and hyper-parameter selection, and leads
  to disqualification. Test images are never released. Attempts to extract them
  from the evaluation container lead to disqualification.
- The use of information about the structure of the hidden test set is not allowed. Using information, e.g., from the image directories to apply separate models/hyperparameters to different subsets, leads to disqualification.
- It is allowed to use the provided validation data to train the final models.

## 4. Submissions

- Submissions are code submissions as described on the Submission page. The
  submitted code must not access the network or tamper with the evaluation
  environment.
- Submission limits per day and in total are shown on the phase.
- The ranking metric is `Challenge_Avg`, see the Evaluation page. The **latest**
  valid submission of a team counts.
- To win, a submission must outperform the organizers' baseline.

## 5. Verification and publication

- The top-ranked teams must provide their training code and a short
  description of the method (fact sheet) so the organizers can verify the
  result and compliance with these terms. Teams that do not comply are removed
  from the final ranking.
- The top three teams release their code under an OSI-approved open-source
  licence.
- All participants are invited to submit papers to the RWS workshop by Oct 23, 2026
  and to present their work at the workshop.

## 6. Data licence

The UPAR annotations are published under the Creative Commons
Attribution-NonCommercial-ShareAlike 3.0 licence. The underlying image datasets
keep their own licences and are for non-commercial research use only. If you
use the data, cite the UPAR paper and the papers of the sub-datasets.

## 7. General conditions

- The organizers may change the schedule, clarify the rules, or disqualify
  participants who violate these terms or the spirit of the challenge.
- The organizers' decisions on ranking and eligibility are final.
- The data and the platform are provided as is, without warranty.

By registering you accept these terms.
