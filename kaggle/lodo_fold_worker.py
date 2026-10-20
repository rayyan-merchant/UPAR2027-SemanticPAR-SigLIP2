"""Independent fold process. Affinity is assigned by the resource scheduler."""
import argparse,gc,json,os,shutil,subprocess,sys
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--held',required=True);ap.add_argument('--feasibility',required=True);args=ap.parse_args()
    base=Path('/kaggle/working');code=base/'lodo_code';suite=base/'lodo';data=suite/'staged_data'
    os.chdir(code);sys.path[:0]=[str(code),str(code/'kaggle')]
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if torch.cuda.device_count()!=1:raise RuntimeError('Each fold must see exactly one assigned GPU')
    from upar.lodo import write_json,digest
    from upar.parity import code_digest
    from upar.data import UPARDataset,label_weights
    from resource_policy import committed_history
    expected=json.loads((code/'reports/lodo/EXPECTED_SPLITS.json').read_text())
    original=suite/'original_evidence';gates=suite/'gates'
    expected_native='55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832'
    write_json(suite/args.held/'AFFINITY.json',{'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'visible_gpu_count':torch.cuda.device_count(),'visible_gpu_name':torch.cuda.get_device_name(),'seed':2027,'workers':2,'torch_threads':2})
    results = []
    for held in [args.held]:
        fold = suite / held; split = fold / 'splits'; trained = fold / 'training'
        write_json(fold / 'WORKER_STATUS.json', {'phase': 'fold', 'held_out': held, 'completed_folds': [r['held_out_domain'] for r in results], 'submitted': False})
        config = {'held_out_domain': held, 'seed': 2027, 'epochs_max': 6, 'head_only_epochs': 1,
            'trainable_upper_vision_blocks': 4, 'microbatch': 16, 'accumulate': 4,
            'backbone_lr': 1e-5, 'head_lr': 1e-4, 'ema_decay': .9995, 'max_patches': 256,
            'foundation_native_sha256': expected_native, 'model_code_sha256': code_digest(),
            'split_hashes': expected[held]['partitions'], 'selection_metric': 'legacy provisional source macro Challenge_Avg; raw/EMA',
            'dataloader_workers': 2, 'source_only_calibration': True, 'production_model_changed': False}
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
            'provenance': 'Prior full image SHA/pixel audit, pinned whole-archive SHA and local ZIP CRC; source partition duplicate checks rerun',
            'cross_partition_decoded_pixel_overlaps': split_receipt['cross_partition_decoded_pixel_overlaps'],
            'original_audit_sha256': digest(original / 'data/data_audit.json')})
        shutil.copyfile(args.feasibility, fold / 'GPU_FEASIBILITY.json')
        gc.collect(); torch.cuda.empty_cache()
        command = [sys.executable, '-m', 'upar.train', '--train-csv', str(split / 'source_train.csv'),
            '--val-csv', str(split / 'source_selection.csv'), '--data-root', str(data), '--output', str(trained),
            '--epochs', '6', '--workers', '2', '--batch-size', '16', '--accumulate', '4',
            '--data-report', str(fold / 'DATA_GATE.json'), '--parity-report', str(gates / 'PRETRAINED_PARITY.json'),
            '--feasibility-report', str(fold / 'GPU_FEASIBILITY.json'), '--max-hours', '3.5']
        if (trained / 'last.pt').exists():
            resume_state = torch.load(trained / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
            if resume_state['schema'] != 2 or resume_state['config']['epochs'] != 6:
                raise ValueError('Not a compatible LODO recovery checkpoint')
            completed = resume_state['epoch'] + 1
            history_path = trained / 'history.jsonl'
            history = committed_history(history_path)
            history = [line for line in history if line['epoch'] <= completed]
            if [line['epoch'] for line in history] != list(range(1, completed + 1)):
                raise ValueError('Committed recovery history is inconsistent')
            history_path.write_text(''.join(json.dumps(line) + '\n' for line in history))
            del resume_state; gc.collect()
            command += ['--resume', str(trained / 'last.pt')]
        write_json(fold/'TRAINING_START.json', {'held_out':held,'training_started':True,'staging_report_sha256':digest(suite/'STAGING.json')})
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
        subprocess.run([sys.executable, '-m', 'upar.lodo', '--fold', str(fold), '--data-root', str(data), '--workers', '2'], check=True)
        results.append(json.loads((fold / 'EVALUATION.json').read_text()))

if __name__=='__main__':main()
