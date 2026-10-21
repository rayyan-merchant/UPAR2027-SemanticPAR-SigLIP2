# Equal-domain sampling comparison — measured results

Locked baseline4b60ffc. One predeclared development comparison; targets were previously inspected. No hidden leaderboard tuning or production replacement.

## Raw target changes

| Target | Baseline | Equal | Delta score | Delta mA | Delta instance-F1 | Delta precision | Delta recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 0.786128 | 0.785534 | -0.000594 | -0.002800 | 0.001961 | 0.001991 | 0.001928 |
| PETA | 0.769342 | 0.765880 | -0.003462 | -0.003473 | -0.003442 | -0.001685 | -0.005122 |
| Market1501 | 0.813343 | 0.808667 | -0.004676 | -0.006833 | -0.002207 | -0.002829 | -0.001613 |

## Calibrated target changes

| Target | Baseline | Equal | Delta score | Delta mA | Delta instance-F1 | Delta precision | Delta recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 0.797104 | 0.797565 | 0.000460 | -0.001236 | 0.002260 | 0.010172 | -0.007072 |
| PETA | 0.777228 | 0.776778 | -0.000451 | 0.001867 | -0.002846 | -0.007077 | 0.001845 |
| Market1501 | 0.817448 | 0.813926 | -0.003522 | -0.005943 | -0.000873 | -0.002047 | 0.000385 |

## Predeclared decision

**retain original square-root-domain sampling**. Criteria met: False.

Macro/worst scores, all metrics and group effects: reports/lodo/equal_domain/results/COMPARISON.json, METRIC_COMPARISON.csv and GROUP_COMPARISON.csv. Actual per-domain draws/unique source images/epoch times: SAMPLING_EXPOSURE.csv. Baseline domain exposure is expected, not measured.

This rule is a conservative development decision, not statistical proof; one seed, source selection, scorer ambiguity and incomplete convergence remain. Actual group/trajectory changes are reviewed below. No further training or competition upload was launched.

## Consolidated comparison

| Setting | Baseline macro | Equal macro | Macro change | Baseline worst | Equal worst | Worst change |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.789605 | 0.786694 | -0.002911 | 0.769342 | 0.765880 | -0.003462 |
| calibrated | 0.797260 | 0.796089 | -0.001171 | 0.777228 | 0.776778 | -0.000451 |

## Interpretation and final sampler decision

**Retain the original square-root-domain sampler. Do not adopt equal-domain sampling for the final model.** Neither macro nor worst-domain transfer improved, so the predeclared +.003/+ .003 adoption margins were not met. PA100K gained only .000460 after calibration, PETA lost .000451 and Market1501 lost .003522. Raw target scores declined in every fold. The calibrated macro decline is .001171 (0.117 percentage points); the worst-domain decline is .000451 (0.045 points). This is mixed/negative evidence, not a convincing generalization improvement. No revised margin or extra sampler setting was searched.

PETA, where PA100K exposure changed from expected73.76% to measured approximately50%, gained .001867 mA after calibration but lost .002847 instance-F1: precision fell .007077 and recall rose .001845. The mA gain did not offset instance-F1 loss. Market1501, where PA100K changed from expected75.12% to approximately50%, lost .005944 mA and .000873 instance-F1; precision fell .002047 while recall rose only .000385. PA100K had a small F1/precision-led calibrated gain despite lower mA/recall, but its baseline exposure was already near50/50. No fold provides evidence of the required broad improvement.

The alternate mean-image-F1 convention also fails the positive macro/worst requirement; see COMPARISON.json for exact values. Source-only calibration still helps each new fold, but its presence does not rescue equal-domain training relative to the calibrated baseline. Attribute-group improvements and regressions coexist; the table below must not be read as permission to choose group-specific thresholds or another model using these targets. PETA improved upper-length and lower-color mA (+.01321/+.01108), but age declined (.01029). Market age mA declined .05153 and hair .01445 despite improved upper-length/accessory groups. The pattern does not support a uniform cross-domain gain.

This experiment is consistent with a tradeoff between more repetition of smaller sources and less exposure/diversity from PA100K, not proof of that causal mechanism. One fixed seed, different source-selected epochs/raw-versus-EMA and finite six-epoch optimization cannot separate systematic sampler effects from training noise. PA and PETA still have their best source checkpoint at epoch6; Market has a descriptive plateau only. The inspected targets are development evidence, not a fresh unbiased test, and no hidden competition information entered the comparison.

## Equal-domain absolute target metrics

| Target | Setting | Challenge_Avg | mA | Instance-F1 | Precision | Recall | Label-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | raw | 0.785534 | 0.756661 | 0.816698 | 0.802739 | 0.831151 | 0.522421 |
| PA100k | calibrated | 0.797565 | 0.785098 | 0.810434 | 0.782528 | 0.840403 | 0.527603 |
| PETA | raw | 0.765880 | 0.742742 | 0.790507 | 0.800338 | 0.780914 | 0.536900 |
| PETA | calibrated | 0.776778 | 0.771586 | 0.782039 | 0.760357 | 0.804995 | 0.538717 |
| Market1501 | raw | 0.808667 | 0.781848 | 0.837392 | 0.847403 | 0.827614 | 0.524411 |
| Market1501 | calibrated | 0.813926 | 0.795027 | 0.833745 | 0.819174 | 0.848845 | 0.525343 |

## Calibrated attribute-group effects

| Target | Group | Baseline mA | Equal mA | Delta mA | Delta label-F1 |
| --- | --- | --- | --- | --- | --- |
| PA100k | age | 0.693444 | 0.689670 | -0.003774 | -0.005355 |
| PA100k | gender | 0.911596 | 0.912112 | +0.000516 | +0.002071 |
| PA100k | hair | 0.804256 | 0.809965 | +0.005710 | -0.000599 |
| PA100k | upper_length | 0.889959 | 0.887933 | -0.002025 | -0.002313 |
| PA100k | upper_colors | 0.813412 | 0.819661 | +0.006249 | +0.007093 |
| PA100k | lower_length | 0.882467 | 0.870826 | -0.011640 | -0.011541 |
| PA100k | lower_colors | 0.759247 | 0.750142 | -0.009105 | +0.004986 |
| PA100k | lower_type | 0.868165 | 0.875053 | +0.006888 | +0.006551 |
| PA100k | accessories | 0.733598 | 0.729279 | -0.004319 | +0.009310 |
| PETA | age | 0.670749 | 0.660464 | -0.010285 | -0.009230 |
| PETA | gender | 0.869989 | 0.869544 | -0.000445 | -0.000597 |
| PETA | hair | 0.833757 | 0.828226 | -0.005531 | +0.000617 |
| PETA | upper_length | 0.875832 | 0.889040 | +0.013208 | +0.000074 |
| PETA | upper_colors | 0.796093 | 0.791436 | -0.004657 | -0.000992 |
| PETA | lower_length | 0.870236 | 0.863529 | -0.006707 | -0.012112 |
| PETA | lower_colors | 0.772342 | 0.783417 | +0.011075 | +0.007515 |
| PETA | lower_type | 0.674816 | 0.683483 | +0.008667 | -0.005438 |
| PETA | accessories | 0.697666 | 0.702010 | +0.004344 | +0.001500 |
| Market1501 | age | 0.775520 | 0.723991 | -0.051530 | +0.006778 |
| Market1501 | gender | 0.915059 | 0.912310 | -0.002749 | -0.003084 |
| Market1501 | hair | 0.796775 | 0.782323 | -0.014452 | -0.014457 |
| Market1501 | upper_length | 0.840400 | 0.853669 | +0.013269 | +0.004068 |
| Market1501 | upper_colors | 0.868329 | 0.867343 | -0.000986 | -0.003736 |
| Market1501 | lower_length | 0.877795 | 0.872580 | -0.005215 | -0.009967 |
| Market1501 | lower_colors | 0.748215 | 0.743801 | -0.004415 | -0.003258 |
| Market1501 | lower_type | 0.912073 | 0.914159 | +0.002086 | +0.021548 |
| Market1501 | accessories | 0.701956 | 0.708959 | +0.007003 | -0.002279 |

Per-attribute raw/calibrated TP/TN/FP/FN/support/P/R/F1 are retained in PER_ATTRIBUTE_TARGET_ERRORS.csv. GROUP_COMPARISON.csv includes both threshold settings. Exact organizer scorer edge cases remain private; mean-image-F1 and unsupported-target-label sensitivity remain in INDEPENDENT_EVALUATION.md and each aggregate EVALUATION.json.

## Source selection, exposure and convergence

| Target excluded | Setting | Baseline source score | Equal source score | Equal source-target gap |
| --- | --- | --- | --- | --- |
| PA100k | raw | 0.841522 | 0.841219 | 0.055685 |
| PA100k | calibrated | 0.850671 | 0.851418 | 0.053853 |
| PETA | raw | 0.849077 | 0.844951 | 0.079071 |
| PETA | calibrated | 0.860265 | 0.854805 | 0.078027 |
| Market1501 | raw | 0.856139 | 0.859042 | 0.050375 |
| Market1501 | calibrated | 0.865920 | 0.867008 | 0.053082 |

| Target excluded | Source domain | Baseline expected fraction | Actual equal draws | Actual fraction | Unique source image rows across6epochs | Available source image rows | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | Market1501 | 0.5179 | 55860 | 0.4987 | 9952 | 10000 | 99.52% |
| PA100k | PETA | 0.4821 | 56148 | 0.5013 | 8653 | 8668 | 99.83% |
| PETA | Market1501 | 0.2624 | 267374 | 0.5007 | 10000 | 10000 | 100.00% |
| PETA | PA100k | 0.7376 | 266632 | 0.4993 | 76303 | 79001 | 96.58% |
| Market1501 | PA100k | 0.7512 | 263288 | 0.5005 | 76265 | 79001 | 96.54% |
| Market1501 | PETA | 0.2488 | 262726 | 0.4995 | 8668 | 8668 | 100.00% |

Unique counts mean distinct source image rows/paths, not person identities or near-duplicate clusters. Baseline actual draws and unique images were not logged; only its expected probabilities are stated. Exact per-epoch draws/unique counts and durations are in SAMPLING_EXPOSURE.csv; aggregate consumption matches all completed training batches. Source-only frequencies, positive/negative counts and weighted BCE values exactly match the locked baseline.

| Target | Baseline best | Equal best | Sampled images | Attempted updates | Successful updates | AMP skips | Epoch compute hours | Convergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 6 raw | 6 raw | 112008 | 1752 | 1752 | 0 | 0.348 | not established |
| PETA | 5 ema | 6 ema | 534006 | 8346 | 8344 | 2 | 1.105 | not established |
| Market1501 | 5 raw | 4 raw | 526014 | 8220 | 8217 | 3 | 1.045 | heuristic plateau |

| Target | Epoch | Baseline loss | Equal loss | Baseline source score | Equal source score | Equal selected weights | Equal epoch seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PA100k | 1 | 0.373816 | 0.376660 | 0.791867 | 0.795805 | raw | 169.1 |
| PA100k | 2 | 0.223183 | 0.222335 | 0.820914 | 0.825183 | raw | 217.9 |
| PA100k | 3 | 0.187926 | 0.191129 | 0.832830 | 0.835908 | raw | 217.5 |
| PA100k | 4 | 0.171036 | 0.174181 | 0.838934 | 0.839280 | raw | 216.2 |
| PA100k | 5 | 0.162454 | 0.161395 | 0.838738 | 0.841103 | raw | 216.4 |
| PA100k | 6 | 0.157667 | 0.158652 | 0.841522 | 0.841219 | raw | 216.6 |
| PETA | 1 | 0.258061 | 0.263137 | 0.831498 | 0.828377 | raw | 480.6 |
| PETA | 2 | 0.165020 | 0.155295 | 0.840584 | 0.841670 | raw | 699.2 |
| PETA | 3 | 0.139557 | 0.114694 | 0.843531 | 0.841075 | raw | 709.2 |
| PETA | 4 | 0.122470 | 0.094304 | 0.846081 | 0.842069 | ema | 696.9 |
| PETA | 5 | 0.113216 | 0.083931 | 0.849077 | 0.844559 | ema | 696.6 |
| PETA | 6 | 0.109612 | 0.079877 | 0.848233 | 0.844951 | ema | 696.6 |
| Market1501 | 1 | 0.262286 | 0.269509 | 0.820309 | 0.829657 | raw | 446.0 |
| Market1501 | 2 | 0.170331 | 0.166054 | 0.840845 | 0.847281 | raw | 665.1 |
| Market1501 | 3 | 0.146127 | 0.131319 | 0.849732 | 0.854186 | raw | 665.1 |
| Market1501 | 4 | 0.130526 | 0.109497 | 0.855407 | 0.859042 | raw | 665.4 |
| Market1501 | 5 | 0.122869 | 0.098902 | 0.856139 | 0.857440 | raw | 662.6 |
| Market1501 | 6 | 0.119624 | 0.094100 | 0.855243 | 0.858051 | ema | 658.2 |

## Actual resources and preserved evidence

The private comparison kernel v1 completed all three six-epoch folds in **1.519GPU-session hours** (5468.32s), versus1.632h baseline; this is not a controlled speed benchmark. Archive staging completed in43.52s, under the120s/30s watchdog. 44 Kaggle tests passed with zero failures/errors. All 41 available runtime source hashes matched local source. Independent source/target metric recomputation, exact source-only calibration refits, split/config/checkpoint locks and exposure audits passed. All31baseline report/evidence hashes remain unchanged.

Kaggle CLI after completion: **19.03h used,10.97h remaining**, displayed precision. This exceeds the8h protected reserve. No active diagnostic compute/download remains after evidence collection; no final training or CodaBench upload has been launched. Diagnostic best/last optimizer/EMA/allRNG/exposure recovery states remain in completed private Kaggle outputs. Weights were not redownloaded: local independent checks mark checkpoint SHA locks remote-only. Private CSV/NPZ/logs remain under outputs/lodo/equal-domain-v1-results, not public Git. Accepted submission1003877 and original D: images remain preserved.

## Final training estimate and deadline plan

**Do not spend GPU hours rerunning the unchanged all-domain recipe solely because this comparison finished.** The accepted model already uses the recommended original sampler. This experiment produces no supported improved production candidate, and the official score remains .770450. LODO models each exclude one domain; they are diagnostics, not replacements to submit. No alternate architecture, interpolation, loss or augmentation change is authorized or established by this result.

If the user later authorizes a justified final run, estimate2.3-3.0h for12epochs with the archive workflow, plus .5-1.0h for calibration/packaging/offline verification; reserve3.5-4.5h with overhead. This is an estimate from measured throughput and the original3.52h pipeline, not a new allocation or guarantee.10.97h remain, but compute availability is not evidence that a rerun would improve the model.

Use the earlier academic targetOctober21 and the user-confirmed **October22,2026 at17:00** cutoff for competition planning. **Asia/Karachi (UTC+5) is the working timezone assumption**, based on this chat; the portal timezone has not been independently verified. Proposed internal schedule, not completed submissions:

| Internal target | Deliverable / decision |
| --- | --- |
| October21,06:00 | Freeze the academic results around accepted1003877, locked LODO baseline and this negative comparison; retain scorer/seed/development-test limitations. |
| October21,10:00 | Finish manuscript/report, reproducibility/code/config/checksum and factsheet draft; obtain real author/affiliation/team/portal/format details and human review. |
| October21,12:00 or earlier portal cutoff | Submit academic materials through the actual portal after human approval/access; preserve a submission receipt. No academic acceptance is currently claimed. |
| October21,18:00 | Competition readiness review: retain accepted model unless a separately authorized, validated candidate exists. Use the user-confirmed October22,17:00 cutoff and confirm portal timezone before the final readiness check. |
| October22,13:00 (four-hour buffer to17:00) | If and only if a new candidate is authorized and verified, start its upload/evaluation with contingency; otherwise keep1003877. Do not use leaderboard search or automatic resubmission. |

The user supplied October22 at5pm. Competition portal timezone and the separate academic cutoff/portal/author/team fields remain unverified; move the internal schedule earlier if those require it. The October21 academic milestones are planning targets from the earlier instruction, not an invented official cutoff. Academic manuscript/portal submission and an improved competition candidate remain separate unfinished deliverables. No organizer email is authorized.
