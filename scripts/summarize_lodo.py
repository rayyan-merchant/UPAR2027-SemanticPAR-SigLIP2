"""Validate saved LODO metrics and write the final report only after all3folds finish."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upar.attributes import ATTRIBUTES
from upar.metrics import macro_score
from upar.data import load_rows
from upar.lodo import digest, GROUPS, write_json
from scripts.generalization_analysis import fit_fast

ORDER = ['PA100k', 'PETA', 'Market1501']


def aggregate(results):
    if [r['held_out_domain'] for r in results] != ORDER:
        raise ValueError('Require all three completed folds in the precommitted order')
    return {setting: {'macro': float(np.mean([r[setting]['legacy']['macro']['Challenge_Avg'] for r in results])),
                     'worst': min(r[setting]['legacy']['macro']['Challenge_Avg'] for r in results),
                     'macro_mean_image_F1_score': float(np.mean([r[setting]['mean_image_f1_sensitivity']['macro']['Challenge_Avg'] for r in results]))}
            for setting in ['target_raw', 'target_calibrated']}


def table(head, rows):
    return '| ' + ' | '.join(head) + ' |\n| ' + ' | '.join(['---'] * len(head)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows) + '\n'


def fmt(value):
    return 'unsupported' if value is None else f'{value:.6f}'


def summarize(artifacts, report, public_directory=None):
    artifacts = Path(artifacts); results = []; checks = []; attribute_rows = []
    public = Path(public_directory) if public_directory else ROOT / 'reports/lodo/results'; public.mkdir(parents=True, exist_ok=True)
    for held in ORDER:
        fold = artifacts / held
        result = json.loads((fold / 'EVALUATION.json').read_text())
        if result['convergence']['completed_epochs'] != 6:
            raise ValueError('Incomplete training: ' + held)
        lock = json.loads((fold / 'LOCK.json').read_text())
        if digest(fold / 'LOCK.json') != result['lock_sha256']:
            raise ValueError('Result lock changed')
        checked = []; remote_only = []
        for remote, expected in lock['files'].items():
            rel = Path(remote.split('/lodo/', 1)[1])
            local = artifacts / rel
            if local.exists():
                if digest(local) != expected: raise ValueError('Locked file mismatch: ' + remote)
                checked.append(str(rel))
            else:
                remote_only.append(rel.as_posix())
        if any(not name.endswith('/training/best.pt') for name in remote_only):
            raise ValueError('Missing non-checkpoint source lock evidence')
        calibration = json.loads((fold / 'calibration.json').read_text())
        if calibration['attributes'] != ATTRIBUTES:
            raise ValueError('Calibration attribute order changed')
        z = np.load(fold / 'target_predictions.npz')
        if digest(fold / 'target_predictions.npz') != result['target_prediction_sha256']:
            raise ValueError('Target predictions changed')
        if set(z['domains']) != {held} or not np.isfinite(z['prob']).all() or not ((z['prob'] >= 0) & (z['prob'] <= 1)).all():
            raise ValueError('Invalid target predictions/domain')
        target_rows = load_rows(fold / 'splits/sealed_target_evaluation.csv')
        if not np.array_equal(z['y'], np.stack([r[1] for r in target_rows])) or not np.array_equal(z['valid'], np.stack([r[2] for r in target_rows])):
            raise ValueError('Target prediction-label rows do not match sealed CSV')
        for setting, thresholds in [('target_raw', np.full(40, .5)), ('target_calibrated', calibration['thresholds'])]:
            actual = macro_score(z['prob'], z['y'], z['valid'], z['domains'].tolist(), thresholds)['macro']
            if any(abs(actual[key] - result[setting]['legacy']['macro'][key]) > 1e-10 for key in actual):
                raise ValueError('Saved target metrics do not reproduce')
            # Independent mean-image-F1 calculation from per-image counts.
            pred = z['prob'] > np.asarray(thresholds); true = z['y'] == 1; valid = z['valid'].astype(bool)
            tp = (pred & true & valid).sum(1); denominator = (pred & valid).sum(1) + (true & valid).sum(1)
            image_f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean()
            expected_f1 = result[setting]['mean_image_f1_sensitivity']['macro']['Inst_F1']
            if abs(image_f1 - expected_f1) > 1e-10: raise ValueError('Mean-image-F1 does not reproduce')
        source_npz = fold / 'source_calibration_predictions.npz'
        if digest(source_npz) != calibration['fit_prediction_sha256']:
            raise ValueError('Source calibration input changed')
        source_z = np.load(source_npz)
        if held in set(source_z['domains']): raise ValueError('Target contaminated source calibration')
        refit = fit_fast(source_z['prob'], source_z['y'], source_z['valid'], source_z['domains'])
        if not np.array_equal(refit, np.asarray(calibration['thresholds'], np.float32)):
            raise ValueError('Source-only threshold fit does not reproduce')
        expected_partitions = json.loads((ROOT / 'reports/lodo/EXPECTED_SPLITS.json').read_text())[held]['partitions']
        for name, evidence in expected_partitions.items():
            if digest(fold / 'splits' / (name + '.csv')) != evidence['sha256']:
                raise ValueError('Split differs from local precommitment')
        checks.append({'held_out': held, 'metric_reproduction_passed': True, 'local_lock_files_checked': checked,
                       'checkpoint_lock_verified_by_remote_evaluation_only': remote_only,
                       'target_prediction_sha256': result['target_prediction_sha256']})
        write_json(public / f'{held}_EVALUATION.json', result)
        per_attribute = json.loads((fold / 'target_attribute_metrics.json').read_text())
        source = result['source_train_supervision']
        for row in per_attribute:
            index = ATTRIBUTES.index(row['attribute'])
            row.update({'held_out': held, 'source_train_positives': source['valid_positives'][index],
                        'source_train_negatives': source['valid_negatives'][index]})
            attribute_rows.append(row)
        results.append(result)
    summary = aggregate(results)
    suite = json.loads((artifacts / 'CONSOLIDATED.json').read_text())
    if abs(suite['macro_target_calibrated'] - summary['target_calibrated']['macro']) > 1e-10:
        raise ValueError('Consolidated macro mismatch')
    write_json(public / 'INDEPENDENT_VERIFICATION.json', {'checks': checks, 'summary': summary})
    with (public / 'PER_ATTRIBUTE_TARGET_ERRORS.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attribute_rows[0])); writer.writeheader(); writer.writerows(attribute_rows)
    detailed = []; comparisons = []; sensitivity_rows = []; convergence_rows = []; support_rows = []
    for r in results:
        held = r['held_out_domain']
        for label in ['raw', 'calibrated']:
            metrics = r['target_' + label]['legacy']['macro']
            detailed.append([held, label] + [fmt(metrics[k]) for k in ['Challenge_Avg', 'mA', 'Inst_F1', 'Inst_Prec', 'Inst_Rec', 'Label_F1']])
            sense = r['target_' + label]['mean_image_f1_sensitivity']['macro']
            sensitivity_rows.append([held, label, fmt(sense['Challenge_Avg']), fmt(sense['harmonic_of_macro_mA_and_mean_image_F1']), fmt(sense['harmonic_all_valid_safe_zero_mA_mean_image_F1'])])
        comparisons.append([held, fmt(r['source']['raw']['legacy']['macro']['Challenge_Avg']),
            fmt(r['source']['calibrated']['legacy']['macro']['Challenge_Avg']), fmt(r['target_raw']['legacy']['macro']['Challenge_Avg']),
            fmt(r['target_calibrated']['legacy']['macro']['Challenge_Avg']), fmt(r['source_minus_target_raw']), fmt(r['source_minus_target_calibrated']), fmt(r['calibration_target_gain'])])
        c = r['convergence']
        convergence_rows.append([held, str(r['best_epoch']) + ' ' + r['best_weights'], r['images_processed_training'],
            r['optimizer_updates_attempted'], r['optimizer_updates_successful'], r['amp_skipped_updates'],
            f"{r['training_epoch_seconds_total'] / 3600:.3f}", fmt(c['selection_score_last_three_range']),
            'heuristic plateau' if c['source_selection_plateau_heuristic'] else 'not established'])
        s = r['source_train_supervision']; pos = np.asarray(s['valid_positives']); neg = np.asarray(s['valid_negatives'])
        support_rows.append([held, ', '.join(ATTRIBUTES[j] for j in np.flatnonzero(pos == 0)) or 'none',
                            ', '.join(ATTRIBUTES[j] for j in np.flatnonzero(neg == 0)) or 'none',
                            ', '.join(ATTRIBUTES[j] for j in np.flatnonzero((pos > 0) & (pos <= 20))) or 'none'])
    repeated = {}
    for group in GROUPS:
        values = [r['target_attribute_groups_calibrated'][group]['mA'] for r in results]
        ranks = [sorted((name for name in GROUPS if r['target_attribute_groups_calibrated'][name]['mA'] is not None),
                        key=lambda name: r['target_attribute_groups_calibrated'][name]['mA'])[:3] for r in results]
        repeated[group] = {'mA_each_fold': values, 'bottom_three_folds': sum(group in rank for rank in ranks)}
    group_table = table(['Group'] + ORDER + ['Bottom3 count'], [[group] + [fmt(v) for v in x['mA_each_fold']] + [x['bottom_three_folds']] for group, x in repeated.items()])
    gains = [r['calibration_target_gain'] for r in results]
    recommendation = '**Pending evidence review.** This script validates and tabulates all three completed folds. An analyst must compare the actual trajectories, support and transfer outcomes and recommend exactly one A–E change; no automatic heuristic chooses or implements a training change.'
    text = '# Three-domain LODO evaluation — completed\n\n'
    text += 'All three six-epoch diagnostics completed from the original foundation. Accepted submission1003877 and the production model/calibration remain unchanged; no competition resubmission occurred.\n\n'
    text += '## Consolidated source-to-target comparison\n\n' + table(['Target', 'Source raw', 'Source calibrated', 'Target raw', 'Target calibrated', 'Raw gap', 'Calibrated gap', 'Target calibration gain'], comparisons)
    text += '\n' + table(['Setting', 'Macro held-out score', 'Worst-domain score', 'Macro mean-image-F1 score'], [[label, fmt(s['macro']), fmt(s['worst']), fmt(s['macro_mean_image_F1_score'])] for label, s in summary.items()])
    text += '\nMacro averages weight the three excluded datasets equally; these are three independently trained models, not a deployable ensemble. Source-selection scores retain checkpoint-selection optimism. Source and target metrics use the same indicated convention.\n\n'
    text += '## Target metrics\n\n' + table(['Target', 'Setting', 'Challenge_Avg', 'mA', 'Instance F1', 'Precision', 'Recall', 'Label F1'], detailed)
    text += '\n## Scorer sensitivity\n\n' + table(['Target', 'Setting', 'Mean image F1 / mean domain H', 'Mean image F1 / H macro', 'Also safe-zero support mA'], sensitivity_rows)
    text += '\nExact private organizer edge cases remain unavailable; neither alternative is asserted to reproduce them. Calibration and checkpoint choice used the original provisional objective and were fixed before target inference.\n\n'
    text += '## Source supervision\n\n' + table(['Target excluded', 'Zero source positives', 'Zero source negatives', 'Source positives1–20'], support_rows)
    text += '\nSupport is measured from source training only. A zero-positive source attribute has no supervised positive examples, and sparse target positives make estimates noisy. This separates inadequate supervision from failures despite abundant source labels; it does not identify a causal visual mechanism.\n\n'
    text += '## Attribute groups\n\n' + group_table
    text += '\nThe full target raw/calibrated TP/TN/FP/FN, P/R/F1/mA and source/target support are in reports/lodo/results/PER_ATTRIBUTE_TARGET_ERRORS.csv. Group mA omits unsupported target attributes; safe-zero sensitivity above exposes that choice.\n\n'
    text += '## Convergence and compute\n\n' + table(['Target', 'Best source checkpoint', 'Sampled images', 'Attempted updates', 'Successful updates', 'AMP skips', 'Epoch compute hours', 'Last3 source range', 'Convergence'], convergence_rows)
    text += f"\nComplete suite elapsed wall time: {suite['elapsed_wall_seconds'] / 3600:.3f}hours, including setup/staging/checkpoint I/O. Epoch timers exclude checkpoint save overhead; per-fold post-training evaluation times are retained in EVALUATION.json. A last3 source range below.002 with an earlier best is only a descriptive plateau heuristic. Six epochs and different update counts cannot establish full convergence or rule out EMA lag.\n\n"
    text += '## Calibration transfer\n\n'
    text += f"Source-only calibration gains are positive in {sum(g > 0 for g in gains)}/3 targets; per-fold gains are " + ', '.join(f'{h}:{g:+.6f}' for h, g in zip(ORDER, gains)) + '. These are untouched target measurements under a fixed checkpoint/threshold lock, not target-fitted offsets.\n\n'
    text += '## Single recommended next change\n\n' + recommendation + '\n\n'
    text += 'No A–E change was implemented by this diagnostic. Domain shift, imbalance and feature specialization can coexist. Differences between this six-epoch protocol and the original twelve-epoch all-domain model prevent attributing the original hidden score gap to one training component. No hidden test identities, images, labels or leaderboard search entered the runs.\n\n'
    text += '## Isolation and reproducibility\n\n'
    text += 'PA CSV bytes were reused exactly; all folds match precommitted split hashes and have zero cross-partition decoded-pixel overlap. Source train, selection and calibration are disjoint, and held-out training rows are excluded entirely. Native foundation hash55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832, fixed seed2027, same SigLIP2NaFlex semantic head/loss/sampler/optimizer/schedule/AMP/EMA/256patches. All six epochs and source-only model/calibration locks were checked. Independent local recomputation reproduced raw/calibrated legacy and mean-image-F1 metrics, calibration-input hashes and available source-lock files. If weights were not downloaded, their server-side checksum lock remains explicitly marked remote-only in INDEPENDENT_VERIFICATION.json. Pixel independence does not prove identity/near-duplicate independence. Checkpoints/logs remain in private Kaggle outputs and local outputs/lodo where retrieved; large weights and sample-level arrays are not published in Git.\n'
    Path(report).write_text(text, encoding='utf-8')
    write_json(public / 'GROUP_REPETITION.json', repeated)
    print('Wrote actual three-fold evaluation:', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--artifacts', required=True)
    parser.add_argument('--report', default=str(ROOT / 'LODO_EVALUATION.md'))
    parser.add_argument('--public-directory', help='Use a separate directory for controlled comparisons; preserve baseline outputs')
    args = parser.parse_args(); summarize(args.artifacts, args.report, args.public_directory)
