"""Prepare a source-only diagnostic split; does not train or inspect hidden data."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def prepare(root, data_root, output, held_out='PA100k', reuse=False, manifest_path=None):
    manifest = json.loads(Path(manifest_path or root / 'reports/data_mapped/image_manifest.json').read_text())
    def read(split):
        path = data_root / f'annotations/task1/{split}/gt.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        return rows[0], rows[1:], hashlib.sha256(path.read_bytes()).hexdigest()
    header, train, train_hash = read('train')
    val_header, val, val_hash = read('val')
    assert header == val_header
    domain = lambda row: row[0].replace('\\', '/').split('/')[0]
    assert held_out in {'PA100k', 'PETA', 'Market1501'}
    source_train = [row for row in train if domain(row) != held_out]
    excluded_train = [row for row in train if domain(row) == held_out]
    source_pixels = {manifest[row[0]]['pixel_sha256'] for row in source_train}
    source_val = [row for row in val if domain(row) != held_out]
    # Remove every decoded-pixel overlap with source train, not just named rows.
    excluded_val = [row for row in source_val if manifest[row[0]]['pixel_sha256'] in source_pixels]
    source_val = [row for row in source_val if manifest[row[0]]['pixel_sha256'] not in source_pixels]
    selection, calibration = [], []
    for row in source_val:
        pixel = manifest[row[0]]['pixel_sha256']
        bucket = int(hashlib.sha256(('lodo2027:' + pixel).encode()).hexdigest()[:12], 16) % 2
        (selection if bucket == 0 else calibration).append(row)
    target = [row for row in val if domain(row) == held_out]
    partitions = {'source_train': source_train, 'source_selection': selection,
                  'source_calibration': calibration, 'sealed_target_evaluation': target}
    pixel_sets = [{manifest[row[0]]['pixel_sha256'] for row in rows} for rows in partitions.values()]
    assert all(not (pixel_sets[i] & pixel_sets[j]) for i in range(4) for j in range(i))
    assert all(domain(row) != held_out for rows in [source_train, selection, calibration] for row in rows)
    output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, rows in partitions.items():
        path = output / (name + '.csv')
        import io
        stream = io.StringIO(newline='')
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
        encoded = stream.getvalue().encode('utf-8')
        if reuse:
            assert path.read_bytes() == encoded, f'Prepared split changed: {path}'
        else:
            path.write_bytes(encoded)
        receipts[name] = {'rows': len(rows), 'domains': dict(Counter(map(domain, rows))),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {'status': 'PREPARED_NOT_EXECUTED', 'held_out_domain': held_out,
              'original_annotation_sha256': {'train': train_hash, 'val': val_hash},
              'excluded_target_train_rows': len(excluded_train),
              'excluded_source_val_train_pixel_duplicates': len(excluded_val),
              'partitions': receipts, 'cross_partition_decoded_pixel_overlaps': 0,
              'pixel_identity_limit': 'Does not rule out same-person/near-duplicate leakage.',
              'initialization': 'Fresh general-purpose foundation revision b53b807d3a2d5e2b3911292f2d69e5341cdc064c; NEVER current PAR-finetuned best/last checkpoint.',
              'precommitted_recipe': {'epochs_max': 6, 'head_only_epochs': 1, 'trainable_vision_blocks': 4,
                 'effective_batch': 64, 'microbatch': 16, 'max_patches': 256, 'seed': 2027,
                 'backbone_lr': 1e-5, 'head_lr': 1e-4, 'ema_decay': .9995,
                 'checkpoint_selection': 'source_selection only, raw/EMA competition-aligned provisional score',
                 'calibration': 'source_calibration only, original global40 coordinate fit; no target calibration',
                 'target_access': 'One evaluation after source checkpoint/thresholds/config hashes locked; no selecting settings by target scores'},
              'execution_gate': 'Require source-only data, original foundation hash, pretrained parity and fold-specific GPU feasibility before training. Target evaluation only after checkpoint/calibration lock.'}
    receipt_path = output / 'LODO_SPLIT_RECEIPT.json'
    if reuse:
        prior = json.loads(receipt_path.read_text())
        assert prior['partitions'] == report['partitions']
        assert prior['original_annotation_sha256'] == report['original_annotation_sha256']
        report['reused_existing_csv_bytes'] = True
    if not reuse:
        receipt_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    prepare(root, Path('D:/UPAR2027/official/data'), root.parent / 'diagnosis_1003877/lodo_plan')
