# Generalization diagnosis — submission 1003877

2026-10-20. CPU diagnosis completed; accepted submission preserved. No new training, modified model, threshold deployment or competition upload was performed. The earlier automatic submission follow-up is paused.

## Finding and decision

The main explanation supported by current evidence is a **validation-protocol/generalization gap**, with a measurable recall/false-positive bias and a smaller scoring-convention discrepancy. A causal attribution to a particular training choice is not yet established. Existing validation contains all three training datasets; it does not estimate performance on unseen surveillance domains.

Official Challenge_Avg is **0.770450** (UI 0.7704). The duplicate-excluded local gap is **9.450 percentage points after calibration**, or **8.573 points before calibration**.

**Highest-value next step:** one fresh-foundation, six-epoch maximum, PA100K-held-out diagnostic using Market1501/PETA only for training, selection and calibration. Keep the accepted competition submission unchanged. Do not start another full all-domain run or choose thresholds/interpolation coefficients from leaderboard feedback.

## 1. Artifact identity and preservation

The actual `Submission File` downloaded from CodaBench submission1003877 is **byte-identical** to the local final ZIP: 358,029,220 bytes, SHA256 `76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308`. This verifies uploaded identity, rather than relying on its filename. Downloads used the existing authenticated file link; signed URLs were not recorded in public reports and the temporary URL file was removed.

Independent retained copies are in `outputs/preserved_1003877/`: scored ZIP, best checkpoint, calibration, validation predictions, history, training configuration, execution log and scoring archive. `PRESERVATION_MANIFEST.json` records hashes; embedded weights/calibration match the preserved standalone artifacts. Canonical last/resume checkpoint remains independently retained.

| Artifact | SHA256 |
| --- | --- |
| Best epoch7 EMA checkpoint | `18a48956eac97d16709e355d57e747a9944c872f844e20645d1dc5aae86ae0d6` |
| Calibration JSON | `84c7a5cd33ca433ceabb9d2014da0b58d85cd5a4c0399bc2b62035cd85c8e12d` |
| Validation predictions | `8c7e126392b32dee9dbf6119f60ff55ac10fa0c2afc176f43e42c5df72d2ff65` |
| Last/resume checkpoint | `64d23096ce17db94f667055c65dfddbf3861844c4e8867e64186ed2eb9259f74` |
| Official scoring archive | `c594a4668b3224178b4c12df560de857d45b7f706ca44de7115e08851ece88b7` |

Training was one twelve-epoch run, selected epoch7 EMA from known-domain validation. Scored ZIP retains the original metadata bytes, native offline model and original batch32 API. Original D: datasets/annotations and accepted model/calibration were not modified. The approved 358MB obsolete candidate-download chunk folder was removed only after hashing it against the complete preserved ZIP; no canonical artifact was removed.

## 2. Legitimate official scoring evidence

Inspected only the four aggregate scoring-result files: `scores.json`, `metadata`, `per_attribute.json`, `detailed_results.html`. No hidden images, labels, per-image predictions or prohibited metadata were retrieved. Public dataset names below come directly from this scoring report; they are descriptive evidence and are not inputs to any fitting, sampling, calibration or model-selection code.

| Metric | Official |
| --- | ---: |
| Challenge_Avg | 0.770450 |
| mA | 0.773647 |
| Instance F1 | 0.767279 |
| Instance precision | 0.741799 |
| Instance recall | 0.807915 |
| Label F1 | 0.583226 |

| Public report domain | mA | Instance precision | Instance recall | Instance F1 | Label F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MEVID | 0.7065 | 0.6744 | 0.7323 | 0.6962 | 0.4442 |
| Synth | 0.8408 | 0.8092 | 0.8835 | 0.8384 | 0.7223 |

The roughly 14.2-point reported instance-F1 difference between these domains is strong evidence of heterogeneous generalization. It does not identify which visual factors or labeling conventions caused the difference. Scores average datasets equally; image-weighted aggregation would be inappropriate. Rounded domain values cannot resolve all scorer aggregation details.

The report states **predict_batch processed all 29,248 images in 109.91 seconds**. Scoring metadata exitCode=0, elapsedTime=1.894s; ingestion elapsed/exit fields are null. HTML scoring diagnostic reports 1.01s. No warning/error text was present in the four files. This is actual accepted-run timing evidence, superseding the earlier T4-only runtime uncertainty; it is not a separate hardware attestation or complete container startup breakdown.

## 3. Validation versus submitted inference parity

`scripts/check_validation_submission_parity.py` actually ran on 48 official validation images (16/domain, deterministic spread), Python3.11.16/torch2.4.1+cpu. Source model/preprocessor/attribute modules are byte-identical to the accepted archive. Independent module namespaces exercised validation `predict` and archived `run.py`.

- All 33,407 stored prediction paths, label arrays, validity masks and domain assignments match the original validation CSV exactly.
- Actual-image preprocessing is identical, including NaFlex RGB patch ordering, 256-token budget, aspect shapes and padding masks. Padding counts in the sample range0–22.
- Native validation versus submitted raw FP32 probability maximum error **0**. Threshold-offset probability maximum error **0**; calibrated decision mismatches **0**. The API applies `sigmoid(logit - logit(threshold))`, so official strict >0.5 implements the stored probability threshold without double calibration.
- Perturbing invalid padded patches in three real images changes logits by **0**.
- Comparing saved T4 torch2.11 FP16-AMP predictions to this torch2.4.1 CPU FP32 run: max probability difference **0.015984**, mean **0.000350**; two raw and one calibrated binary disagreements among1,920 cells. This is a bounded sample observation, not a whole-validation numerical error bound.
- Previously retained real trained offline API tests checked exact40 attribute ordering, finite probabilities, original metadata, single/batch consistency, batching35 and blocked network/HF/timm/requests imports. Real pretrained native/reference FP32 and FP16 layer parity also passed on four aspect ratios.

No evidence currently supports a deployment-path, checkpoint, label-order or padding bug as the main cause. Full hidden inference parity is unavailable and was not attempted. New evidence: `validation_submission_parity.json`.

## 4. Scoring mismatch and its measured size

Current provisional local scorer computes **H(mean image precision, mean image recall)** for instance F1. Official public per-domain numbers do not satisfy this identity: MEVID H(.6744,.7323)≈.70216 vs reported .6962; Synth H(.8092,.8835)≈.84472 vs .8384. Rounding cannot explain those differences. Thus at least one convention differs. **Mean per-image F1 is a plausible alternative, not a verified reconstruction of the private scorer.**

Current local mA also excludes attributes with no positive or no negative support in a domain. Market validation has six zero-positive attributes: upper brown/orange/pink and lower orange/purple/red. Organizer treatment is not exposed by this archive.

| Local sensitivity, duplicate-excluded | Raw | Accepted calibrated |
| --- | ---: | ---: |
| Existing provisional score | 0.856180 | 0.864952 |
| Mean per-image F1, mean domain harmonics | 0.854590 | 0.863077 |
| Mean per-image F1, harmonic of domain-macro components | 0.854618 | 0.863150 |
| Also include zero-support labels using zero safe divisions (unverified alternative) | 0.846134 | 0.854553 |

Mean-image-F1 convention changes the calibrated score by about **0.19 points**. The joint zero-support sensitivity changes it by **1.04 points**, still leaving about **8.41 points** to the official score. These are scenarios, not an additive or exhaustive decomposition of the hidden gap. Do not silently overwrite the accepted scoring implementation or claim exact organizer parity. Any future diagnostic should log both F1 conventions, support counts and both macro aggregation orders; organizer-source differential testing remains unavailable. No organizer contact was sent.

## 5. Calibration robustness on allowed validation data

Reproduced all40 original thresholds exactly with a faster sufficient-statistic implementation, checked against the untouched reference on a fixture and the full fit. Five folds use seed2027 SHA256 of decoded-pixel hashes to keep duplicate pixels together, approximately balancing domains. Excluded known PETA train/validation duplicates08573/08574 from all calibration assessments:33,405 rows remain.

| Fold | Evaluation rows | Raw score | Disjoint calibrated score |
| --- | --- | --- | --- |
| 0 | 6668 | 0.858220 | 0.864129 |
| 1 | 6834 | 0.851431 | 0.860792 |
| 2 | 6656 | 0.857673 | 0.863293 |
| 3 | 6615 | 0.857144 | 0.863589 |
| 4 | 6632 | 0.860859 | 0.868689 |

Pooled out-of-fold score is **0.863482**, versus fitted-on-all **0.864952**: optimism **0.147 points**, retained OOF improvement over raw **0.730 points**. All five folds improved under the existing provisional objective. Threshold ranges span more than0.10 across folds for10attributes, so rare-attribute thresholds remain unstable despite aggregate robustness. Thirty-three thresholds are below0.5, five above, two equal; sixteen hit the lower grid bound0.22. This is evidence of objective pressure, not authorization to extend the grid.

**Limitation:** the fixed checkpoint had already been selected using the full validation split. These folds estimate calibration optimism conditional on that checkpoint, not unbiased checkpoint selection or unseen-domain generalization. Pixel grouping does not prove identity/near-duplicate independence. Removing the two exact cross-split duplicates changes raw score only0.00000483 (~0.00048points), too little to explain the observed gap; other leakage remains unmeasured.

Domain-aware threshold transfer fits only the other two known validation datasets:

| Calibration-held domain | Raw | Fit other two domains | Fit all three (in-sample) |
| --- | --- | --- | --- |
| Market1501 | 0.848187 | 0.852834 | 0.853922 |
| PA100k | 0.845154 | 0.854135 | 0.858075 |
| PETA | 0.875198 | 0.878296 | 0.882858 |

Transfer improves all three known domains, but less than fitting on them. **This is calibration-domain exclusion, not model LODO:** the model trained on every dataset. Neither this table nor five-fold calibration establishes robustness to unseen surveillance domains.

## 6. False-positive bias and weakest attributes

| Domain | FP raw → calibrated | FN raw → calibrated | Instance precision raw → calibrated | Instance recall raw → calibrated |
| --- | --- | --- | --- | --- |
| Market1501 | 16641 → 19221 | 15355 → 14234 | 0.8745 → 0.8609 | 0.8816 → 0.8904 |
| PA100k | 10439 → 12941 | 8066 → 6855 | 0.8516 → 0.8299 | 0.8791 → 0.8975 |
| PETA | 5626 → 7141 | 4675 → 4183 | 0.8864 → 0.8655 | 0.9021 → 0.9124 |

Across known validation domains, calibration increases FP **32,706→39,303 (+20.2%)**, while reducing FN **28,096→25,272 (−10.1%)**. It raises mA and recall while reducing instance precision/F1 in each domain. Macro precision changes .870830→.852097, recall .887599→.900097, F1 .879114→.875367. The official P<R pattern is consistent with recall bias, but its aggregate report does not expose hidden confusion counts, so no hidden FP/FN totals or causal claim is made.

For weighted BCE, `wp/wn=exp(1−2p)`. At an ideal population optimum under the **sampled training distribution**, `logit(q)=logit(η)+(1−2p)`. Thus positive weighting already lowers the effective positive decision boundary for rare attributes; a calibrated threshold below0.5 pushes it further. For AgeYoung (training prevalence .046637), q-threshold .22 corresponds theoretically to η≈.102, not .22. This is explanatory math, not a claim that network outputs are calibrated posteriors. Pooled training prevalence also differs from the domain-balanced sampler's effective prevalence. No unweighted-loss training ablation has been run, so weighting cannot be assigned a measured share of the test gap.

Group summaries average attributes and known domains; unsupported positive/negative mA cells are excluded locally. These are descriptive group means, not a new competition objective.

| Group | Known raw mA | Known calibrated mA | Known calibrated label F1 | Official mA | Official label F1 |
| --- | --- | --- | --- | --- | --- |
| age | 0.773675 | 0.804113 | 0.686348 | 0.591529 | 0.454474 |
| gender | 0.933124 | 0.933229 | 0.923429 | 0.888404 | 0.868074 |
| hair | 0.841758 | 0.861665 | 0.742847 | 0.787850 | 0.706793 |
| upper_length | 0.860921 | 0.864350 | 0.917410 | 0.878380 | 0.828244 |
| upper_colors | 0.864916 | 0.883416 | 0.626198 | 0.815331 | 0.629106 |
| lower_length | 0.933687 | 0.943004 | 0.871244 | 0.888395 | 0.758750 |
| lower_colors | 0.815364 | 0.846126 | 0.524687 | 0.753921 | 0.500229 |
| lower_type | 0.887250 | 0.902495 | 0.848805 | 0.836595 | 0.812590 |
| accessories | 0.779675 | 0.787218 | 0.605447 | 0.729673 | 0.442593 |

Known-domain weaknesses are accessories/age in mA and lower clothing colors in label F1. Official aggregate age performance is substantially weaker (mA .591529); accessories/other-color categories also fail. Most striking: **AgeYoung official label F1=0**, mA .497936, although its threshold is already .22. Known calibrated AgeYoung has39/92/237 positives in Market/PA/PETA and recall .974/.478/.595 respectively. This supports a rare/ambiguous-age domain-transfer problem; the aggregate archive cannot distinguish absent support, missed positives, label semantics, image quality or other causes. Do not infer hidden class counts or choose new age thresholds from this result.

AgeAdult illustrates why F1 alone misleads: official F1 .902216 but mA .583718. LowerColorOther official mA/F1 .533213/.122379, UpperColorOther .553899/.190657, Bag .676662/.404167 and Sunglasses .678694/.182739 are other weak aggregates. Strong upper red/yellow and gender performance argues against a uniform output permutation, although it cannot prove every prediction is correct. Full40-attribute tables and per-domain TP/TN/FP/FN/support/P/R/F1/mA are retained, without hidden-label inference.

## 7. Ranked causes, evidence and limits

| Rank | Candidate cause | Evidence | What remains unproven |
| --- | --- | --- | --- |
| 1 | Same-dataset validation overstates unseen-domain performance | All three datasets occur in training/validation; official domains differ strongly; age/accessory deterioration | Specific appearance/semantic mechanisms and actual source-held-out gap require fresh LODO |
| 2 | Recall/FP bias from weighted BCE plus low thresholds | FP +20.2% on known validation, precision down in all3,16thresholds at lower bound | No hidden confusion counts or unweighted training control; calibration still helps known-domain OOF |
| 3 | Local scoring conventions inflate comparison | Official P/R/F1 identity mismatch; measured local F1/zero-support sensitivity0.19–1.04points | Exact private scorer edge cases/aggregation not recoverable from aggregates |
| 4 | Fine-tuned upper layers/head specialize to source domains | Best epoch7 vs epoch12 EMA raw .852531; backbone blocks8–11 moved | No held-out-domain comparison proving feature drift harmful or interpolation helpful |
| 5 | Deployment/numeric mismatch | Actual48-image parity, correct artifact/hash/ordering/offsets, accepted runtime | Only bounded known-data inference compared; no complete FP32-vs-AMP validation rerun |
| 6 | Known exact train/val duplicate leakage | Two excluded validation rows, raw score effect~0.00048points | Person-level and near-duplicate overlap not audited; exact duplicate effect negligible |

## 8. Minimal true leave-one-domain-out diagnostic (prepared, not run)

Hold out **all PA100K**: chosen because removing the largest source yields the cheapest informative leave-one-dataset-out experiment, using allowed known-data counts. This choice is not based on hidden dataset names, images, labels or a search against leaderboard scores.

| Partition | Rows | Domains |
| --- | --- | --- |
| source_train | 18668 | Market1501:10000, PETA:8668 |
| source_selection | 11634 | Market1501:8202, PETA:3432 |
| source_calibration | 11785 | Market1501:8256, PETA:3529 |
| sealed_target_evaluation | 9986 | PA100k:9986 |

All79,001 PA training rows are excluded, including prevalence, sampling, warmup, finetuning, selection and calibration. The9,986 official PA validation rows are a sealed known-data target evaluation. Source validation removes two source-train pixel duplicates and splits decoded-pixel groups into selection/calibration; actual cross-partition pixel overlaps **0**. Original annotations stay read-only; derived CSV hashes/receipts are local in `outputs/diagnosis_1003877/lodo_plan/`.

Initialize from the original general-purpose foundation and text queries, **not current PAR best/last weights**, which have already seen PA. Same architecture/augmentations/loss, one head-only epoch then upper four vision blocks, max6epochs, LR1e-5/1e-4, batch16×4,256patches,EMA.9995,seed2027. Recompute prevalence and domain sampler only from18,668 source-train rows. Select raw/EMA checkpoint on source-selection rows only; fit one global40threshold vector on source-calibration rows only. Lock checkpoint/config/calibration hashes before one PA evaluation. Record raw/calibrated metrics with both F1 conventions, per-attribute support and macro-domain versus target performance. Do not select epochs/settings using target performance.

Measured T4 rates imply six source epochs ≈803s training; evaluating both raw/EMA on11,634 selection rows each epoch ≈716s; source-calibration and target inference ≈112s. Nominal compute subtotal≈27minutes. Budget **0.5–1GPUhour**, conservatively **1.25h wall-time cap once running**, plus variable queue/data staging/setup and checkpoint I/O. This extrapolation from short training and local-input inference measurements is not a benchmark of the pilot. At the fresh read-only quota check26.01GPUhours remained; actual allocation is unconfirmed and earlier verification queued about90minutes. Allow~2GB persistent checkpoint space and derived-private data storage; C: had only~0.29GB free after the narrowly approved cleanup, so use Kaggle working storage and bounded streaming downloads rather than another local weight copy.

No new GPU job was launched because the user requested findings/recommendation before substantial additional compute. The split-generation script and receipt are complete. A runnable training job still needs source-specific audit/feasibility gate receipts and separate selection/calibration handling; do not bypass existing hash gates or run unchanged all-domain `upar.train` against this design. Six epochs are a diagnostic, not a promised competitive final model; undertraining and a single held-out dataset limit conclusions.

## 9. Pretrained/fine-tuned weight interpolation

[WiSE-FT (Wortsman et al., CVPR2022)](https://arxiv.org/abs/2109.01903) and its [official implementation](https://github.com/mlfoundations/wise-ft) provide legitimate evidence for weight interpolation between compatible zero-shot and fine-tuned classifiers under distribution shift. They do not establish gains for this PAR head.

Read-only tensor audit: frozen vision blocks0–7 are exactly unchanged. Blocks8–11 relative L2 changes are .007425/.005898/.004919/.007275 (about0.49–0.74%); other backbone tensors aggregate .000094. Small parameter distance is not a bound on functional/domain drift. `finetune_drift.json` contains complete results. No interpolated state or predictions were generated.

The foundation-native checkpoint's PAR classifier is random/untrained, although its semantic queries are initialized. **Naive full-model WiSE-FT would mix a trained PAR head with an untrained classifier**, not an established zero-shot PAR solution. A backbone-only interpolation retaining the trained head is a testable hypothesis with feature/head mismatch risk. It cannot be approved from current same-source validation or official score feedback.

Only after a legitimate source-held-out baseline: either preregister one coefficient (e.g.0.5) as a diagnostic comparison without target-driven coefficient selection, or select coefficients/calibration in a nested source-only domain split and evaluate once on an untouched outer target. Nested source training adds roughly another pilot's compute and may lack representative rare labels. No deployment/resubmission or interpolation implementation is justified yet.

## 10. Candidate actions and decision

| Action | Value | Resource / risk | Decision |
| --- | --- | --- | --- |
| Fresh PA-held-out pilot | Directly tests main validation-protocol hypothesis |0.5–1GPUh estimate, queue uncertainty,6epoch undertraining | **Single recommended next step; prepared, not launched** |
| Log scorer sensitivities/support on all future validation | Removes a known measurement ambiguity | CPU seconds; exact organizer scorer still private | Diagnostic implementation complete; accepted scorer/model unchanged |
| Change threshold regularization/loss | May rebalance FP/recall | Requires legitimate source-held-out control; could lower mA/recall | Do not deploy from this leaderboard outcome |
| Backbone interpolation | Low inference cost if eventually validated | Source-held-out/nested calibration needed; head mismatch | No modified model created |
| Another full all-domain run / architecture variant | Does not isolate gap cause | Hours and weak evidence | Do not launch |

The report supports a ranked diagnosis, not an exact causal decomposition or a promised >.81 score. Preserve submission1003877. No automatic resubmission, hidden-data adaptation or leaderboard hyperparameter search is authorized.

## Evidence and reproduction

- `scripts/generalization_analysis.py`: CPU full-validation sufficient-statistic calibration parity,5fold assessment, domain calibration transfer, confusion/group reports; measured65.59CPUelapsed seconds on the original completed run.
- `scripts/check_validation_submission_parity.py`: actual48-image source/API inference audit, reuses a hash-verified extraction to avoid duplicate weights.
- `scripts/prepare_lodo_diagnosis.py`: source/target-derived CSVs and hashes only; no training.
- `scripts/inspect_finetune_drift.py`: read-only backbone tensor distances, no interpolation.
- New diagnostic math tests: fast-fit parity with invalid/zero-support labels and explicit difference between the two F1 definitions. **Full local suite22passed in53.52s.** An initial unrestricted pytest invocation collected a stale ignored extracted offline script; pytest.ini now confines discovery to tests/, and a fresh full rerun passed. Original model/inference/loss/calibrator are unchanged.
- Public aggregate evidence mirrors are in `reports/generalization_1003877/`. Full official archive, server ZIP, preservation receipt, fold identities and derived CSVs remain local under `outputs/`; no images, weights or private sample-level arrays are committed.

## Appendix: all40 attribute aggregates

Known values below are domain means under the accepted calibration, after excluding the two duplicate validation rows. mA omits a domain's attribute if it lacks positive or negative support; precision/recall/F1 use safe divisions. Official values are the scoring archive's dataset means, not recomputed from hidden labels. For exact support/confusions and raw-versus-calibrated changes, use `per_attribute_known_domains.csv`.

| Attribute | Known mA | Known precision | Known recall | Known F1 | Official mA | Official F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Age-Young | 0.834662 | 0.392075 | 0.682519 | 0.479903 | 0.497936 | 0.000000 |
| Age-Adult | 0.783650 | 0.984594 | 0.982607 | 0.983586 | 0.583718 | 0.902216 |
| Age-Old | 0.794028 | 0.601415 | 0.596430 | 0.595554 | 0.692932 | 0.461207 |
| Gender-Female | 0.933229 | 0.922253 | 0.924851 | 0.923429 | 0.888404 | 0.868074 |
| Hair-Length-Short | 0.889078 | 0.917677 | 0.956088 | 0.936453 | 0.768874 | 0.822070 |
| Hair-Length-Long | 0.916971 | 0.846197 | 0.899523 | 0.871705 | 0.863263 | 0.813544 |
| Hair-Length-Bald | 0.778946 | 0.334403 | 0.572315 | 0.420382 | 0.731413 | 0.484765 |
| UpperBody-Length-Short | 0.864350 | 0.929249 | 0.906627 | 0.917410 | 0.878380 | 0.828244 |
| UpperBody-Color-Black | 0.922468 | 0.834229 | 0.923264 | 0.876342 | 0.824900 | 0.783941 |
| UpperBody-Color-Blue | 0.883233 | 0.656313 | 0.804283 | 0.719940 | 0.794808 | 0.649955 |
| UpperBody-Color-Brown | 0.874270 | 0.297307 | 0.525470 | 0.355789 | 0.744617 | 0.532598 |
| UpperBody-Color-Green | 0.912519 | 0.604986 | 0.840350 | 0.693315 | 0.750542 | 0.553407 |
| UpperBody-Color-Grey | 0.842028 | 0.686425 | 0.756505 | 0.718619 | 0.781423 | 0.549102 |
| UpperBody-Color-Orange | 0.904501 | 0.384082 | 0.543876 | 0.450070 | 0.934179 | 0.788416 |
| UpperBody-Color-Pink | 0.883961 | 0.308401 | 0.525141 | 0.387765 | 0.859335 | 0.632155 |
| UpperBody-Color-Purple | 0.867383 | 0.479295 | 0.755000 | 0.581328 | 0.790065 | 0.540517 |
| UpperBody-Color-Red | 0.951383 | 0.700792 | 0.923063 | 0.784322 | 0.954111 | 0.862813 |
| UpperBody-Color-White | 0.912068 | 0.798287 | 0.884127 | 0.837680 | 0.828375 | 0.643913 |
| UpperBody-Color-Yellow | 0.922641 | 0.611009 | 0.857085 | 0.694981 | 0.967724 | 0.821802 |
| UpperBody-Color-Other | 0.716488 | 0.368088 | 0.475829 | 0.414222 | 0.553899 | 0.190657 |
| LowerBody-Length-Short | 0.943004 | 0.823769 | 0.929071 | 0.871244 | 0.888395 | 0.758750 |
| LowerBody-Color-Black | 0.888569 | 0.874776 | 0.911489 | 0.892482 | 0.797181 | 0.712396 |
| LowerBody-Color-Blue | 0.852668 | 0.687581 | 0.777608 | 0.725918 | 0.791218 | 0.726396 |
| LowerBody-Color-Brown | 0.885172 | 0.454918 | 0.805114 | 0.549728 | 0.764890 | 0.519247 |
| LowerBody-Color-Green | 0.838087 | 0.479864 | 0.682663 | 0.555580 | 0.705621 | 0.347308 |
| LowerBody-Color-Grey | 0.821631 | 0.633853 | 0.733946 | 0.680166 | 0.711591 | 0.448643 |
| LowerBody-Color-Orange | 0.841851 | 0.405093 | 0.456410 | 0.408765 | 0.721036 | 0.432432 |
| LowerBody-Color-Pink | 0.849680 | 0.451216 | 0.707190 | 0.545537 | 0.872803 | 0.602405 |
| LowerBody-Color-Purple | 0.725288 | 0.180952 | 0.301587 | 0.226190 | 0.728683 | 0.556627 |
| LowerBody-Color-Red | 0.934958 | 0.339793 | 0.585761 | 0.419333 | 0.874958 | 0.526627 |
| LowerBody-Color-White | 0.904082 | 0.477285 | 0.847994 | 0.599993 | 0.796621 | 0.508285 |
| LowerBody-Color-Yellow | 0.799895 | 0.292777 | 0.604167 | 0.346905 | 0.749235 | 0.500000 |
| LowerBody-Color-Other | 0.837400 | 0.308273 | 0.693968 | 0.345641 | 0.533213 | 0.122379 |
| LowerBody-Type-Trousers&Shorts | 0.906032 | 0.985208 | 0.973739 | 0.979436 | 0.775190 | 0.981488 |
| LowerBody-Type-Skirt&Dress | 0.898958 | 0.627285 | 0.842836 | 0.718174 | 0.897999 | 0.643692 |
| Accessory-Backpack | 0.862419 | 0.763338 | 0.784561 | 0.773274 | 0.770149 | 0.490980 |
| Accessory-Bag | 0.805915 | 0.740314 | 0.720936 | 0.730203 | 0.676662 | 0.404167 |
| Accessory-Glasses-Normal | 0.760907 | 0.561649 | 0.594438 | 0.563044 | 0.734031 | 0.482916 |
| Accessory-Glasses-Sun | 0.647347 | 0.285823 | 0.313541 | 0.274072 | 0.678694 | 0.182739 |
| Accessory-Hat | 0.859501 | 0.645870 | 0.737062 | 0.686642 | 0.788829 | 0.652163 |
