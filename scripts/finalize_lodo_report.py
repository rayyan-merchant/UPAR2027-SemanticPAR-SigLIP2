"""Add reviewed aggregate analysis to the independently verified threefold report.

Run summarize_lodo.py first. This reads only permitted source/target diagnostic
outputs; it never trains, changes thresholds, or accesses competition results.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import numpy as np


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows) + '\n'


def main(repo, artifacts):
    repo, artifacts = Path(repo), Path(artifacts)
    sys.path.insert(0, str(repo))
    from upar.metrics import macro_score
    from upar.data import load_rows
    from upar.lodo import digest, group_report
    order = ['PA100k', 'PETA', 'Market1501']
    public = repo / 'reports/lodo/results'
    results = [json.loads((artifacts / h / 'EVALUATION.json').read_text()) for h in order]
    extra_checks, curves, errors, weak, source_groups, probes = [], [], [], [], [], []
    for h, r in zip(order, results):
        fold = artifacts / h
        z = np.load(fold / 'training/best_val_predictions.npz')
        rows = load_rows(fold / 'splits/source_selection.csv')
        assert h not in set(z['domains'])
        assert np.array_equal(z['y'], np.stack([x[1] for x in rows]))
        assert np.array_equal(z['valid'], np.stack([x[2] for x in rows]))
        calibration = json.loads((fold / 'calibration.json').read_text())
        for name, threshold in [('raw', np.full(40, .5)), ('calibrated', calibration['thresholds'])]:
            actual = macro_score(z['prob'], z['y'], z['valid'], z['domains'].tolist(), threshold)['macro']
            assert all(abs(actual[k] - r['source'][name]['legacy']['macro'][k]) < 1e-10 for k in actual)
        extra_checks.append({'held_out': h, 'source_selection_labels_and_metrics_reproduced': True,
                             'best_validation_prediction_sha256': digest(fold / 'training/best_val_predictions.npz')})
        hist = [json.loads(x) for x in (fold / 'training/history.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in hist] == list(range(1, 7))
        for x in hist:
            curves.append([h, x['epoch'], f"{x['train_loss']:.6f}", f"{x['metrics']['macro']['Challenge_Avg']:.6f}", x['selected'], f"{x['epoch_seconds']:.1f}"])
        attrs = json.loads((fold / 'target_attribute_metrics.json').read_text())
        s = r['source_train_supervision']
        for setting in ['target_raw', 'target_calibrated']:
            a = [x for x in attrs if x['setting'] == setting]
            errors.append([h, setting.removeprefix('target_'), sum(x['tp'] for x in a), sum(x['tn'] for x in a), sum(x['fp'] for x in a), sum(x['fn'] for x in a)])
        for x in sorted([x for x in attrs if x['setting'] == 'target_calibrated' and x['mA'] is not None], key=lambda x: x['mA'])[:6]:
            i = s['attributes'].index(x['attribute'])
            weak.append([h, x['attribute'], s['valid_positives'][i], x['positives'], f"{x['mA']:.4f}", f"{x['precision']:.4f}", f"{x['recall']:.4f}", f"{x['f1']:.4f}"])
        # Each source-domain group is summarized separately before an equal-domain mean.
        source_attrs = json.loads((fold / 'source_attribute_metrics.json').read_text())
        domains = sorted({x['domain'] for x in source_attrs})
        by_domain = [group_report([x for x in source_attrs if x['domain'] == d]) for d in domains]
        for g, target in r['target_attribute_groups_calibrated'].items():
            vals = [x[g]['mA'] for x in by_domain if x[g]['mA'] is not None]
            mean = float(np.mean(vals))
            source_groups.append([h, g, mean, target['mA'], mean - target['mA']])
    for f in sorted((artifacts / 'resource').glob('*.json')):
        if 'SYSTEM' in f.name:
            continue
        p = json.loads(f.read_text())
        probes.append([f.stem, f"{p['images_per_second']:.2f}", f"{p['loader_images_per_second']:.2f}", f"{p['peak_allocated_gb']:.3f}", p['CUDA_VISIBLE_DEVICES']])
    (public / 'SOURCE_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(extra_checks, indent=2) + '\n')
    with (public / 'GROUP_SOURCE_TARGET_COMPARISON.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['held_out', 'group', 'source_equal_domain_mA', 'target_mA', 'gap']); w.writerows(source_groups)
    recommendation = '''**Recommend D: test equal probability per source domain in the existing sampler.** Change only the domain exposure, from the present square-root domain-size mass to 50/50 in a two-source fold (and one-third each in a later all-domain model). Keep architecture, augmentation, weighted BCE, optimizer, schedule, EMA, preprocessing and seed fixed. This is a recommendation for a controlled comparison, not a demonstrated improvement or an authorization to train/deploy it.

The observed sampler assigns PA100K 73.76% of draws in the PETA-held-out fold and 75.12% in the Market-held-out fold. Both have substantial source-target gaps despite over 8,200 updates. Age and accessories are bottom-three groups in all three targets, including failures with thousands of positive source examples. Domain-balanced exposure directly addresses a measured source imbalance without introducing a new architecture or changing calibration, which already transfers positively in all three folds. The PA-held-out sampler is already near 50/50 and still has a gap: sampling imbalance cannot be the whole explanation, and this experiment does not prove D will outperform A/B/C.

**Highest-value next step:** one predeclared, source-only controlled repeat of the same threefold protocol with equal-domain sampling and the same six-epoch budgets, compared against these locked baselines. Retain a further disjoint diagnostic check set if feasible; the present target results have now been inspected and cannot be described as untouched for later method development. Do not search multiple exponents or choose a fold-specific setting. Judge the fixed proposal on macro/worst-domain transfer, with the same scorer sensitivity and no hidden leaderboard feedback. A single repeat is estimated at 1.6–2.6 GPU-session hours (allow 3 hours), based on this suite and its conservative projection. Oversampling smaller sources may slow data loading or overfit repeated examples; retain the throughput/budget watchdogs.

Only if the controlled comparison supports it should an all-domain candidate be trained from the original foundation. A 12-epoch all-domain run is estimated at 2.3–3.0 session hours plus 0.5–1.0 hours for calibration, packaging and offline verification, from current measured image throughput and validation/checkpoint overhead. Combined planning allowance is roughly 5–7 hours; estimates are not reservations or guarantees. The live quota check after completion reports **12.50 hours remaining**, leaving headroom. No further GPU job, modified model or competition upload was launched during this report.

Leaving the accepted model unchanged has a verified official score and no regression risk. D is more promising as a *next experiment* because it changes a measured exposure skew at low implementation cost; no measured result yet establishes that replacing the accepted model is preferable. Keep submission1003877 as the production baseline until a new candidate has legitimate validation and API evidence.'''
    report = repo / 'LODO_EVALUATION.md'
    text = report.read_text(encoding='utf-8')
    pending = '**Pending evidence review.** This script validates and tabulates all three completed folds. An analyst must compare the actual trajectories, support and transfer outcomes and recommend exactly one A–E change; no automatic heuristic chooses or implements a training change.'
    assert pending in text
    text = text.replace(pending, recommendation)
    analysis = '''## Interpretation and evidence ranking

1. **Measured cross-domain transfer weakness is the leading finding.** Source-to-target calibrated gaps are 5.36 points for PA100K, 8.30 for PETA and 4.85 for Market1501. The worst raw held-out score (PETA, .76934) is near the original official .77045, showing that permitted source-domain shifts can produce a gap of comparable scale. This is not evidence that PETA identifies the hidden domain or that the datasets have equivalent difficulty. The hidden gap cannot be decomposed causally from these runs.
2. **Attribute imbalance and label-distribution/semantic shift contribute plausible, distinct limitations.** All 40 attributes have both positive and negative source examples in every fold. PA-held-out lower-orange has only four positive source examples; its zero target recall is consistent with inadequate rare-label supervision. Other failures persist with substantial supervision: Market-held-out normal glasses has 13,038 source positives but calibrated target recall .129. Such failures cannot be explained by missing source labels alone. Age/accessory groups repeatedly transfer poorly; rare colors and heterogeneous 'Other' categories also fail. Annotation-policy differences versus visual feature shift are not separately identifiable here.
3. **Calibration is a secondary tradeoff, not the dominant demonstrated cause.** It improves held-out Challenge_Avg on all three targets by .41–1.10 points and macro by .77 points, while increasing FP and reducing FN. Instance-F1 and label-F1 need not improve when the mA-focused objective improves. Some attributes worsen (Market normal glasses recall falls from .206 to .129); there is no universal FP-only or FN-only failure. No target thresholds were refitted.
4. **Pretrained representation degradation and the best fine-tuning strategy remain untested mechanisms.** There is no frozen-backbone or pretrained/fine-tuned interpolation counterfactual. Source improvement alongside a target gap does not establish that visual representations were damaged. A/B may help, but claiming that these runs prove them would be unsupported. No interpolation was implemented or deployed.

The threefold target macro is .79726 after calibration, still below the roughly .865 known-domain validation score. This is not an apples-to-apples estimate of the hidden-score gap: training domains, examples, updates, six versus twelve epochs, raw/EMA selection and scorer conventions differ. PA's best is at epoch6 and its source score still improves; its smaller update budget and EMA lag are especially important. PETA is not demonstrably converged; Market shows only a descriptive plateau. AMP skipped 0/2/3 updates, under .04% per fold; no infrastructure failure or missing fold is indicated.

| Candidate family | Evidence and decision |
| --- | --- |
| A: preserve pretrained representations | No counterfactual measurement; plausible hypothesis only. |
| B: change fine-tuning | Unequal updates/EMA and incomplete convergence confound the mechanism; not selected now. |
| C: attribute imbalance | Clear rare-label limitations, but abundant-label transfer failures rule out a complete explanation. Retain current loss for the proposed comparison. |
| D: domain sampling | Measured 74–75% large-source exposure and repeated transfer gaps; selected as one bounded testable change. No simultaneous augmentation change. |
| E: calibration stability | Positive aggregate transfer in all3; per-attribute regressions merit reporting, but replacing calibration is lower priority. |

## Error balance and weakest individual attributes

'''
    analysis += table(['Target', 'Setting', 'TP', 'TN', 'FP', 'FN'], errors)
    analysis += '\nCounts pool valid attribute decisions within a target; they are not instance-average scores.\n\n'
    analysis += table(['Target', 'Attribute', 'Source positives', 'Target positives', 'mA', 'Precision', 'Recall', 'F1'], weak)
    analysis += '\nSix weakest supported target attributes per fold by calibrated mA. Sparse target positives have high estimation uncertainty; zero-positive Market color attributes are covered by the separate scorer sensitivity, not ranked as measured positive recall.\n\n'
    analysis += '## Actual source-only trajectories\n\n' + table(['Target excluded', 'Epoch', 'Training loss', 'Selection score', 'Selected weights', 'Epoch seconds'], curves)
    analysis += '\nSource/group comparison is in reports/lodo/results/GROUP_SOURCE_TARGET_COMPARISON.csv. Source groups average the two source datasets equally; unsupported attributes can differ across domains, so group gaps are descriptive.\n\n'
    analysis += '## Staging, concurrency and preservation receipt\n\n'
    stage = json.loads((artifacts / 'STAGING.json').read_text())
    analysis += f"GPU-session archive staging verified all131076 paths in **{stage['elapsed_seconds']:.2f}s**, under120s/30s limits; no mounted per-image copy/hash/decode loop. CPU preflight was34.42s. Source-only concurrent probes retained92.8% of PA solo throughput and95.7% of PETA solo throughput. Separate CUDA affinity0/1, workers2 and threads2 were used; Market reusedGPU0 after PA.\n\n"
    analysis += table(['Probe', 'Training images/s', 'CPU loader images/s', 'Peak allocated GiB', 'Physical GPU'], probes)
    analysis += '\nThese are48-batch source probes, not epoch throughput or deployment timing. Full six-epoch timings above include validation; startup/checkpoint overhead is included in the1.632-hour suite wall time. Provider quota changed from14.132 to12.50 remaining, consistent with approximately1.63 billed session hours, not twice the wall time.38 Kaggle tests passed, zero failed/skipped. Actual Torch2.11/CUDA12.8 runtime matched reused parity evidence; this is not a new torch2.4 GPU parity test.\n\n'
    analysis += 'Canonical submission1003877 ZIP/best/last/calibration remain preserved. Diagnostic best/last recovery checkpoints remain in the completed private Kaggle kernel outputs; local independent verification checks their server-bound lock but does not claim to rehash undownloaded weights. Small logs/configs/CSV/NPZ are retained privately in outputs/lodo/threefold-v2-results. Public Git contains aggregate metrics, curves, configs and reproducibility receipts, not images, weights, sample predictions or annotation CSVs.\n\n'
    text = text.replace('## Single recommended next change', analysis + '## Single recommended next change')
    report.write_text(text, encoding='utf-8')
    receipt = json.loads((repo / 'reports/lodo/THREEFOLD_SUBMISSION_RECEIPT.json').read_text())
    receipt.update(status='COMPLETE', all_folds_six_epochs_verified=True, independent_metrics_and_source_calibration_verified=True,
                   solo_and_concurrent_gpu_gates='PASSED_PARALLEL', elapsed_wall_seconds=5874.516162106,
                   gpu_staging_seconds=stage['elapsed_seconds'], kaggle_tests_passed=38,
                   gpu_quota_remaining_hours_displayed=12.50, gpu_quota_used_hours_displayed=17.50,
                   results_directory='../lodo/threefold-v2-results', recommendation='D: test equal-domain sampling; not implemented')
    from datetime import datetime, timezone
    receipt['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    (repo / 'reports/lodo/THREEFOLD_SUBMISSION_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    current = '''# Status — three-domain LODO complete, 2026-10-20

All three fresh-foundation diagnostic folds completed six epochs; independent
metric recomputation, source-only threshold refits and split/config/calibration
locks passed. See LODO_EVALUATION.md for the reviewed analysis and limitations.

| Held out | Raw Challenge_Avg | Source-calibrated target |
| --- | ---: | ---: |
| PA100K | .786128 | .797104 |
| PETA | .769342 | .777228 |
| Market1501 | .813343 | .817448 |
| Macro | .789605 | .797260 |
| Worst | .769342 | .777228 |

Private kernel rayyannmerchant/upar2027-threefold-archives-resource-safe v1
COMPLETE, runtime source e994492. Independent parallel processes;1.632h wall,
131076 images archive-staged32.96s,38 Kaggle tests passed.12.50GPUh remain.
No active diagnostic compute or download remains. Accepted submission1003877
and its model/ZIP/calibration are unchanged. No final retraining or upload.

Recommendation D is one controlled equal-domain sampling comparison, not a
proven improvement or an instruction to deploy. Calibration improved all3
held-out scores. Representation degradation is unproven; unequal update counts,
incomplete source convergence and sparse labels limit causal interpretation.

Best/last diagnostics remain private in completed Kaggle outputs. Small complete
artifacts are local under outputs/lodo/threefold-v2-results; public reports are
reports/lodo/results. All original D: data remain read-only.

## Preserved production evidence and earlier history

'''
    old = (repo / 'STATUS.md').read_text(encoding='utf-8')
    marker = '- CodaBench submission **1003877'
    (repo / 'STATUS.md').write_text(current + old[old.index(marker):], encoding='utf-8')
    for name in ['STAGING.json','RESOURCE_DECISION.json','INITIALIZATION_RECEIPT.json','SUITE_RECEIPT.json','tests.xml']:
        shutil.copyfile(artifacts / name, public / name)
    for h in order:
        dest=public/h;dest.mkdir(exist_ok=True)
        for name in ['FOLD_CONFIG.json','SOURCE_SUPERVISION.json','AFFINITY.json']:
            shutil.copyfile(artifacts/h/name, dest/name)
        shutil.copyfile(artifacts/h/'training/history.jsonl', dest/'history.jsonl')
    (public/'TRAINING_CURVES.csv').write_text('held_out,epoch,training_loss,selection_score,selected_weights,epoch_seconds\n'+'\n'.join(','.join(map(str,x)) for x in curves)+'\n')
    repro=repo/'REPRODUCIBILITY.md'
    with repro.open('a',encoding='utf-8') as f:
        f.write('\n\n## Completed archive-based threefold diagnostics\n\nRuntime e994492; private kernel upar2027-threefold-archives-resource-safe v1. See LODO_EVALUATION.md and reports/lodo/THREEFOLD_SUBMISSION_RECEIPT.json. Reproduce independent analysis with `python scripts/summarize_lodo.py --artifacts ../lodo/threefold-v2-results/lodo`, then `python scripts/finalize_lodo_report.py --repo . --artifacts ../lodo/threefold-v2-results/lodo`. The second command applies the reviewed recommendation; neither command trains or accesses competition data. Full private artifacts are required; weights remain server-locked and remote-only.\n')
    print('Reviewed report and aggregate receipts written; no training or submission performed.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--artifacts',required=True)
    a=p.parse_args();main(a.repo,a.artifacts)
