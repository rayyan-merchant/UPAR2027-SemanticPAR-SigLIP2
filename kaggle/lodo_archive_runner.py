"""PA-held-out only; verified opaque archives, unchanged model/training recipe; no submission."""
import concurrent.futures
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    base = Path('/kaggle/working'); code = base / 'lodo_code'
    os.chdir(code); sys.path.insert(0, str(code))
    from upar.lodo import digest, write_json
    suite = base / 'lodo'; suite.mkdir(exist_ok=True)
    expected = json.loads((code / 'reports/lodo/EXPECTED_SPLITS.json').read_text())
    original_native = []; private_data = []; recoveries = []
    # Inventory roots only; never enumerate/download all-domain PAR checkpoints.
    for current, dirs, files in os.walk('/kaggle/input'):
        current = Path(current)
        if 'ARCHIVES.json' in files and (current / 'Market1501_PAheld.upar').is_file():
            private_data.append(current)
        if 'foundation_native.pt' in files and (current / 'PRETRAINED_PARITY.json').is_file():
            original_native.append(current)
        if 'SUITE_RECEIPT.json' in files and current.name == 'lodo':
            recoveries.append(current); dirs.clear()
        dirs[:] = [d for d in dirs if d not in {'runs', 'submission_candidate', 'hf-cache', 'upar_code'}]
    if len(private_data) != 1 or len(original_native) != 1 or len(recoveries) > 1:
        raise RuntimeError('Require exactly one official private data source and one original native foundation source')
    source_data = private_data[0]; original = original_native[0]
    started = time.monotonic()
    write_json(base / 'STATUS.json', {'phase': 'preflight', 'training_completed': False, 'submitted': False})
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.1',
                    'tokenizers>=0.22,<0.23', 'huggingface_hub>=0.34,<1', 'pytest', 'Pillow>=10'], check=True)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU allocated; refusing CPU training')
    original_hardware = json.loads((original.parent / 'hardware.json').read_text())
    if torch.__version__ != original_hardware['torch'] or torch.version.cuda != original_hardware['cuda']:
        raise RuntimeError('GPU runtime differs from real-weight parity evidence; rerun source-only reference parity before training')
    free = shutil.disk_usage(base).free
    if free < 8 * 1024**3:
        raise RuntimeError('Need at least 8GiB free for data, three best/last states and atomic saves')
    write_json(base / 'hardware.json', {'python': sys.version, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'gpu_count': torch.cuda.device_count(),
        'working_disk_free_bytes': free, 'vram_bytes': torch.cuda.get_device_properties(0).total_memory})
    (base / 'requirements-executed.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    write_json(base / 'source_hashes.json', {str(p.relative_to(code)): digest(p) for p in code.rglob('*.py')})
    from upar.parity import code_digest
    parity = json.loads((original / 'PRETRAINED_PARITY.json').read_text())
    if not parity['passed'] or not parity['mixed_precision_checked'] or parity['code_sha256'] != code_digest():
        raise ValueError('Original native implementation parity is missing or stale')
    expected_native = '55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832'
    if parity['prepared_sha256'] != expected_native or digest(original / 'foundation_native.pt') != expected_native:
        raise ValueError('Original untrained native foundation checkpoint hash mismatch')
    old_audit = json.loads((original / 'data/data_audit.json').read_text())
    if not old_audit['passed'] or old_audit['leakage_review_required']:
        raise ValueError('Original complete data audit not approved')
    if recoveries:
        # Recovery source is an explicitly attached earlier LODO job, never the all-domain training state.
        shutil.copytree(recoveries[0], suite, dirs_exist_ok=True, ignore=shutil.ignore_patterns('staged_data'))
    if not (suite / 'SUITE_RECEIPT.json').exists():
        write_json(suite / 'SUITE_RECEIPT.json', {'completed_folds': [], 'all_folds_complete': False, 'submitted': False})
    gates = suite / 'gates'; gates.mkdir(exist_ok=True)
    shutil.copy2(original / 'foundation_native.pt', gates / 'foundation_native.pt')
    shutil.copy2(original / 'PRETRAINED_PARITY.json', gates / 'PRETRAINED_PARITY.json')
    foundation = torch.load(gates / 'foundation_native.pt', map_location='cpu', weights_only=False)
    if foundation.get('trained', False) or foundation['attributes'] != __import__('upar.attributes', fromlist=['ATTRIBUTES']).ATTRIBUTES:
        raise ValueError('PAR-trained initialization is prohibited')
    del foundation; gc.collect()
    write_json(suite / 'INITIALIZATION_RECEIPT.json', {'native_sha256': expected_native,
        'origin': 'Original general-purpose HF foundation converted before any PAR training; includes pretrained text query initialization and untrained PAR head',
        'foundation_revision': 'b53b807d3a2d5e2b3911292f2d69e5341cdc064c',
        'parity_code_sha256': code_digest(), 'historical_gpu_parity_reused': True,
        'note': 'Prior implementation QA only; no new target model inference is used before source locks. Source-only GPU feasibility reruns per fold.',
        'all_domain_PAR_checkpoint_opened': False})
    from scripts.prepare_lodo_diagnosis import prepare
    index = suite / 'annotation_index'
    for split_name in ['train','val']:
        annotation = source_data / ('annotations_' + split_name + '.csv')
        if digest(annotation) != expected['PA100k']['original_annotation_sha256'][split_name]: raise ValueError('Annotation bytes changed')
        target = index / 'annotations/task1' / split_name / 'gt.csv'
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(annotation,target)
    for held in ['PA100k']:
        fold = suite / held; split = fold / 'splits'
        receipt = prepare(code, index, split, held, manifest_path=original / 'data/image_manifest.json')
        if receipt['partitions'] != expected[held]['partitions'] or receipt['original_annotation_sha256'] != expected[held]['original_annotation_sha256']:
            raise ValueError('Mounted-data split differs from locally audited precommitment: ' + held)
    # Mounted files are three opaque archives, not 131076 individual image reads.
    from upar.data import load_rows, label_weights, UPARDataset
    sys.path.insert(0, str(code / 'kaggle'))
    from archive_stage import stage
    archive_info = json.loads((source_data / 'ARCHIVES.json').read_text())
    expected_archive_sha = {'Market1501_PAheld.upar': '707e083cdefa972462b37f4358d63ce75e95f538c04131cc034ed6008481db72', 'PETA_PAheld.upar': '0c5f2a07f2822c63238448975c51c8c4ee9b42e738ca38175a6470263eb72891', 'PA100k_PAheld.upar': 'c71254af305f4a9e1b39b6addb2e24e141e37b6c037d330677cedea81484fbf6'}
    wanted = {'Market1501_PAheld.upar','PETA_PAheld.upar','PA100k_PAheld.upar'}
    chosen = [dict(x, path=str(source_data / x['archive'])) for x in archive_info['archives'] if x['archive'] in wanted]
    if {x['archive'] for x in chosen} != wanted: raise RuntimeError('Missing selected fold archive')
    if {x['archive']:x['archive_sha256'] for x in chosen} != expected_archive_sha: raise RuntimeError('Archive provenance changed')
    required = {r[0] for p in (suite/'PA100k/splits').glob('*.csv') for r in load_rows(p)}
    data = suite / 'staged_data'
    write_json(base/'STATUS.json', {'phase':'stream_local_archives','images_needed':len(required),'training_started':False})
    staging = stage(chosen, data, suite/'STAGING.json', max_seconds=120, max_idle=30, max_projected=120)
    if staging['images'] != len(required): raise RuntimeError('Archive union count differs from sealed splits')
    # Membership check is local; no mounted per-image metadata requests.
    if any(not (data / name).is_file() for name in required): raise RuntimeError('Missing required image')
    gc.collect()
    # Synthetic and real-source-image software tests, not an accuracy experiment.
    first_source = load_rows(suite / 'PA100k/splits/source_train.csv')[0][0]
    os.environ['UPAR_TEST_IMAGE'] = str(data / first_source)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', '--junitxml=' + str(suite / 'tests.xml')], check=True)
    results = []
    for held in ['PA100k']:
        fold = suite / held; split = fold / 'splits'; trained = fold / 'training'
        write_json(base / 'STATUS.json', {'phase': 'fold', 'held_out': held, 'completed_folds': [r['held_out_domain'] for r in results], 'submitted': False})
        config = {'held_out_domain': held, 'seed': 2027, 'epochs_max': 6, 'head_only_epochs': 1,
            'trainable_upper_vision_blocks': 4, 'microbatch': 16, 'accumulate': 4,
            'backbone_lr': 1e-5, 'head_lr': 1e-4, 'ema_decay': .9995, 'max_patches': 256,
            'foundation_native_sha256': expected_native, 'model_code_sha256': code_digest(),
            'split_hashes': expected[held]['partitions'], 'selection_metric': 'legacy provisional source macro Challenge_Avg; raw/EMA',
            'source_only_calibration': True, 'production_model_changed': False}
        if (fold / 'FOLD_CONFIG.json').exists() and json.loads((fold / 'FOLD_CONFIG.json').read_text()) != config:
            raise ValueError('Recovery configuration differs from locked fold')
        write_json(fold / 'FOLD_CONFIG.json', config)
        if (fold / 'EVALUATION.json').exists():
            from upar.lodo import verify_lock
            verify_lock(json.loads((fold / 'LOCK.json').read_text()))
            results.append(json.loads((fold / 'EVALUATION.json').read_text())); continue
        source_ds = UPARDataset(split / 'source_train.csv', data, 256)
        wp, wn, prevalence = label_weights(source_ds)
        yy = np.stack([r[1] for r in source_ds.rows]); vv = np.stack([r[2] for r in source_ds.rows])
        from collections import Counter
        counts = Counter(r[3] for r in source_ds.rows); mass = {k: v**.5 / sum(x**.5 for x in counts.values()) for k, v in counts.items()}
        attrs = __import__('upar.attributes', fromlist=['ATTRIBUTES']).ATTRIBUTES
        supervision = {'attributes': attrs, 'valid_positives': ((yy == 1) & vv).sum(0).tolist(),
           'valid_negatives': ((yy == 0) & vv).sum(0).tolist(), 'valid_counts': vv.sum(0).tolist(),
           'training_prevalence': prevalence.tolist(), 'positive_weights': wp.tolist(), 'negative_weights': wn.tolist(),
           'domain_counts': dict(counts), 'expected_sampling_domain_mass': mass,
           'source_train_only': True, 'target_domain_absent': held not in counts}
        write_json(fold / 'SOURCE_SUPERVISION.json', supervision)
        del source_ds, yy, vv; gc.collect()
        split_receipt = json.loads((split / 'LODO_SPLIT_RECEIPT.json').read_text())
        write_json(fold / 'DATA_GATE.json', {'passed': True, 'leakage_review_required': False,
            'splits': {'train': {'sha256': digest(split / 'source_train.csv')}, 'val': {'sha256': digest(split / 'source_selection.csv')}},
            'provenance': 'Original full decode/pixel audit plus all-byte-hash-verified staging; source partition duplicate checks rerun',
            'cross_partition_decoded_pixel_overlaps': split_receipt['cross_partition_decoded_pixel_overlaps'],
            'original_audit_sha256': digest(original / 'data/data_audit.json')})
        from upar.feasibility import benchmark
        benchmark(str(split / 'source_train.csv'), str(data), str(gates / 'PRETRAINED_PARITY.json'), str(fold / 'GPU_FEASIBILITY.json'), batch_size=16)
        gc.collect(); torch.cuda.empty_cache()
        command = [sys.executable, '-m', 'upar.train', '--train-csv', str(split / 'source_train.csv'),
            '--val-csv', str(split / 'source_selection.csv'), '--data-root', str(data), '--output', str(trained),
            '--epochs', '6', '--workers', '4', '--batch-size', '16', '--accumulate', '4',
            '--data-report', str(fold / 'DATA_GATE.json'), '--parity-report', str(gates / 'PRETRAINED_PARITY.json'),
            '--feasibility-report', str(fold / 'GPU_FEASIBILITY.json'), '--max-hours', '1.5' if held == 'PA100k' else '2.5']
        if (trained / 'last.pt').exists():
            resume_state = torch.load(trained / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
            if resume_state['schema'] != 2 or resume_state['config']['epochs'] != 6:
                raise ValueError('Not a compatible LODO recovery checkpoint')
            completed = resume_state['epoch'] + 1
            history_path = trained / 'history.jsonl'
            history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
            history = [line for line in history if line['epoch'] <= completed]
            if [line['epoch'] for line in history] != list(range(1, completed + 1)):
                raise ValueError('Committed recovery history is inconsistent')
            history_path.write_text(''.join(json.dumps(line) + '\n' for line in history))
            del resume_state; gc.collect()
            command += ['--resume', str(trained / 'last.pt')]
        write_json(base/'TRAINING_START.json', {'held_out':held,'training_started':True,'staging_report_sha256':digest(suite/'STAGING.json')})
        with (fold / 'training_execution.log').open('a', buffering=1) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); print(held, line.rstrip(), flush=True)
            if process.wait() != 0:
                raise RuntimeError('Training failed; per-fold best/last and logs retained: ' + held)
        actual_prevalence = json.loads((trained / 'prevalence.json').read_text())
        if any(actual_prevalence[name] != prevalence[j] for j, name in enumerate(attrs)):
            raise ValueError('Training prevalence differs from source-only audit')
        # Separate process releases optimizer state/VRAM before calibration/target inference.
        subprocess.run([sys.executable, '-m', 'upar.lodo', '--fold', str(fold), '--data-root', str(data)], check=True)
        results.append(json.loads((fold / 'EVALUATION.json').read_text()))
        write_json(suite / 'SUITE_RECEIPT.json', {'completed_folds': [r['held_out_domain'] for r in results], 'all_folds_complete': False, 'selected_plan_complete': len(results) == 1, 'submitted': False})
    write_json(suite / 'CONSOLIDATED.json', {'folds': results,
        'aggregation_scope': 'PA100k only; NOT three-domain macro',
        'macro_target_raw': float(np.mean([r['target_raw']['legacy']['macro']['Challenge_Avg'] for r in results])),
        'macro_target_calibrated': float(np.mean([r['target_calibrated']['legacy']['macro']['Challenge_Avg'] for r in results])),
        'worst_target_raw': min(r['target_raw']['legacy']['macro']['Challenge_Avg'] for r in results),
        'worst_target_calibrated': min(r['target_calibrated']['legacy']['macro']['Challenge_Avg'] for r in results),
        'elapsed_wall_seconds': time.monotonic() - started, 'new_competition_submissions': 0})
    write_json(base / 'STATUS.json', {'phase': 'complete', 'completed_folds': [r['held_out_domain'] for r in results], 'submitted': False})
    # Derived staging only, never originals or checkpoints. Avoid persisting another image copy.
    if data.resolve().is_relative_to(base) and data.name == 'staged_data':
        shutil.rmtree(data)


if __name__ == '__main__':
    main()
