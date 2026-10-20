"""Source-only checkpoint/calibration lock, followed by a single fixed-target evaluation.

Uses the unchanged training recipe in upar.train. Never reads CodaBench data.
AMP/checkpoint patterns are inherited from train.py and documented at:
https://docs.pytorch.org/docs/2.4/notes/amp_examples.html
https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .attributes import ATTRIBUTES
from .data import UPARDataset
from .model import UPARModel, VisionConfig
from .train import predict
from .parity import code_digest
from scripts.generalization_analysis import fit_fast, sensitivity, attribute_rows


GROUPS = {'age': range(3), 'gender': [3], 'hair': [4, 5, 6], 'upper_length': [7],
          'upper_colors': range(8, 20), 'lower_length': [20], 'lower_colors': range(21, 33),
          'lower_type': [33, 34], 'accessories': range(35, 40)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def lock_files(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def verify_lock(lock):
    for path, expected in lock['files'].items():
        if digest(path) != expected:
            raise ValueError('Locked source checkpoint/config/calibration changed: ' + path)


def report_predictions(p, y, v, domains, thresholds):
    return sensitivity(p, y, v, np.asarray(domains), np.asarray(thresholds, np.float32))


def group_report(attribute_results):
    result = {}
    for name, indices in GROUPS.items():
        attrs = {ATTRIBUTES[i] for i in indices}
        selected = [r for r in attribute_results if r['attribute'] in attrs]
        ma = [r['mA'] for r in selected if r['mA'] is not None]
        result[name] = {'mA': None if not ma else float(np.mean(ma)),
                        'label_F1': float(np.mean([r['f1'] for r in selected])),
                        'attributes_with_zero_positive_support': [r['attribute'] for r in selected if r['positives'] == 0]}
    return result


def convergence(history):
    scores = [h['metrics']['macro']['Challenge_Avg'] for h in history]
    losses = [h['train_loss'] for h in history]
    best = int(np.argmax(scores)) + 1
    late_range = float(np.ptp(scores[-3:]))
    loss_drop = float((losses[-3] - losses[-1]) / max(abs(losses[-3]), 1e-12)) if len(losses) >= 3 else None
    return {'completed_epochs': len(history), 'source_best_epoch': best,
            'selection_score_last_three_range': late_range, 'training_loss_last_three_relative_drop': loss_drop,
            'best_at_final_epoch': best == len(history),
            'source_selection_plateau_heuristic': len(history) >= 6 and late_range < .002 and best < len(history),
            'interpretation': 'Descriptive source-only heuristic, not proof of convergence. Six epochs, small-source updates and EMA lag can limit conclusions; no target-based extension.'}


def evaluate_fold(fold, data_root, workers=4):
    fold = Path(fold); split = fold / 'splits'; trained = fold / 'training'
    config = json.loads((fold / 'FOLD_CONFIG.json').read_text())
    history = [json.loads(line) for line in (trained / 'history.jsonl').read_text().splitlines() if line]
    if len(history) != 6 or history[-1]['epoch'] != 6:
        raise RuntimeError('Resume to six completed epochs before opening target evaluation')
    result_path = fold / 'EVALUATION.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text()); verify_lock(json.loads((fold / 'LOCK.json').read_text()))
        return saved
    start = time.perf_counter()
    ck = torch.load(trained / 'best.pt', map_location='cpu', weights_only=False)
    if not ck.get('trained') or ck['attributes'] != ATTRIBUTES or ck['parity_code_sha256'] != code_digest():
        raise ValueError('Selected checkpoint provenance mismatch')
    model = UPARModel(VisionConfig(**ck['vision_config'])); model.load_state_dict(ck['state_dict'], strict=True)
    model.cuda().eval()
    lock_path = fold / 'LOCK.json'
    if lock_path.exists():
        # Recovery reuses the fixed source calibration, never refits against target.
        lock = json.loads(lock_path.read_text()); verify_lock(lock)
        thresholds = np.asarray(json.loads((fold / 'calibration.json').read_text())['thresholds'], np.float32)
        source_report = json.loads((fold / 'SOURCE_EVALUATION.json').read_text())
    else:
        if (fold / 'target_predictions.npz').exists():
            raise ValueError('Unbound target predictions exist before source lock')
        calibration = UPARDataset(split / 'source_calibration.csv', data_root, 256, augment=False)
        if any(row[3] == config['held_out_domain'] for row in calibration.rows):
            raise ValueError('Held-out domain present in calibration')
        cal_loader = DataLoader(calibration, batch_size=32, num_workers=workers, shuffle=False, pin_memory=True)
        p, y, v, d = predict(model, cal_loader, torch.device('cuda'))
        thresholds = fit_fast(p, y, v, np.asarray(d))
        np.savez_compressed(fold / 'source_calibration_predictions.npz', prob=p, y=y, valid=v, domains=d)
        calibration_artifact = {'attributes': ATTRIBUTES, 'thresholds': thresholds.tolist(),
            'fit_partition_sha256': digest(split / 'source_calibration.csv'),
            'fit_prediction_sha256': digest(fold / 'source_calibration_predictions.npz'),
            'checkpoint_sha256': digest(trained / 'best.pt'), 'objective': 'original provisional dataset-macro Challenge_Avg, reg=.005, two coordinate passes',
            'global_vector': True, 'target_used': False}
        write_json(fold / 'calibration.json', calibration_artifact)
        z = np.load(trained / 'best_val_predictions.npz')
        source_report = {'raw': report_predictions(z['prob'], z['y'], z['valid'], z['domains'], np.full(40, .5)),
                         'calibrated': report_predictions(z['prob'], z['y'], z['valid'], z['domains'], thresholds),
                         'calibration_fit_in_sample': report_predictions(p, y, v, d, thresholds),
                         'note': 'Selection labels chose checkpoint; calibration labels are disjoint. Selection metrics retain checkpoint-selection optimism.'}
        source_attributes = attribute_rows(z['prob'], z['y'], z['valid'], z['domains'], thresholds, 'source_selection_calibrated')
        write_json(fold / 'source_attribute_metrics.json', source_attributes)
        write_json(fold / 'SOURCE_EVALUATION.json', source_report)
        fixed_files = [trained / 'best.pt', fold / 'calibration.json', fold / 'FOLD_CONFIG.json', fold / 'SOURCE_EVALUATION.json']
        fixed_files += sorted(split.glob('*.csv')) + [split / 'LODO_SPLIT_RECEIPT.json']
        lock = {'held_out_domain': config['held_out_domain'], 'files': lock_files(fixed_files),
                'selection': 'source_selection raw/EMA provisional macro Challenge_Avg only',
                'calibration': 'source_calibration only', 'target_predictions_exist_at_lock': False,
                'training_complete_epochs': 6, 'timestamp_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
        write_json(lock_path, lock)
    verify_lock(lock)
    # First target model inference occurs only below the immutable source lock.
    target_prediction_path = fold / 'target_predictions.npz'
    if target_prediction_path.exists():
        z = np.load(target_prediction_path); p, y, v, d = (z[k] for k in ['prob', 'y', 'valid', 'domains'])
    else:
        target = UPARDataset(split / 'sealed_target_evaluation.csv', data_root, 256, augment=False)
        if any(row[3] != config['held_out_domain'] for row in target.rows):
            raise ValueError('Target contains a source domain')
        target_loader = DataLoader(target, batch_size=32, num_workers=workers, shuffle=False, pin_memory=True)
        p, y, v, d = predict(model, target_loader, torch.device('cuda'))
        tmp = fold / 'target_predictions.tmp.npz'
        np.savez_compressed(tmp, prob=p, y=y, valid=v, domains=d)
        tmp.replace(target_prediction_path)
    target_raw = report_predictions(p, y, v, d, np.full(40, .5))
    target_cal = report_predictions(p, y, v, d, thresholds)
    raw_attributes = attribute_rows(p, y, v, np.asarray(d), np.full(40, .5), 'target_raw')
    cal_attributes = attribute_rows(p, y, v, np.asarray(d), thresholds, 'target_calibrated')
    write_json(fold / 'target_attribute_metrics.json', raw_attributes + cal_attributes)
    support = json.loads((fold / 'SOURCE_SUPERVISION.json').read_text())
    source_score = source_report['raw']['legacy']['macro']['Challenge_Avg']
    result = {'held_out_domain': config['held_out_domain'], 'best_epoch': ck['epoch'] + 1,
       'best_weights': ck['selected'], 'source': source_report, 'target_raw': target_raw, 'target_calibrated': target_cal,
       'source_minus_target_raw': source_score - target_raw['legacy']['macro']['Challenge_Avg'],
       'source_minus_target_calibrated': source_report['calibrated']['legacy']['macro']['Challenge_Avg'] - target_cal['legacy']['macro']['Challenge_Avg'],
       'calibration_target_gain': target_cal['legacy']['macro']['Challenge_Avg'] - target_raw['legacy']['macro']['Challenge_Avg'],
       'target_attribute_groups_raw': group_report(raw_attributes), 'target_attribute_groups_calibrated': group_report(cal_attributes),
       'source_train_supervision': support, 'convergence': convergence(history),
       'images_processed_training': history[-1]['images_processed_cumulative'],
       'optimizer_updates_attempted': history[-1]['optimizer_updates_attempted_cumulative'],
       'optimizer_updates_successful': sum(h['optimizer_updates_successful_epoch'] for h in history),
       'amp_skipped_updates': sum(h['amp_skipped_updates_epoch'] for h in history),
       'training_epoch_seconds_total': sum(h['epoch_seconds'] for h in history),
       'post_training_evaluation_seconds': time.perf_counter() - start,
       'target_prediction_sha256': digest(target_prediction_path), 'lock_sha256': digest(lock_path),
       'target_unique_locked_configuration_evaluations': 1,
       'scorer_limitations': 'Both legacy harmonic-of-mean-P/R and mean-image-F1 plus zero-support/macro-order sensitivities retained. Exact organizer scorer private.'}
    verify_lock(lock); write_json(result_path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--fold', required=True)
    parser.add_argument('--data-root', required=True); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); print(json.dumps(evaluate_fold(args.fold, args.data_root, args.workers), indent=2), flush=True)
