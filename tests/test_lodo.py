import csv
import json
from pathlib import Path
import pytest
from scripts.prepare_lodo_diagnosis import prepare
from upar.attributes import ATTRIBUTES
from upar.lodo import lock_files, verify_lock, convergence
from scripts.summarize_lodo import aggregate


@pytest.mark.parametrize('held', ['PA100k', 'PETA', 'Market1501'])
def test_fold_isolation_and_reuse(tmp_path, held):
    root = tmp_path / 'code'; data = tmp_path / 'data'; output = tmp_path / 'split'
    (root / 'reports/data_mapped').mkdir(parents=True)
    manifest = {}; train = []; val = []
    for domain in ['PA100k', 'PETA', 'Market1501']:
        for i in range(4):
            name = f'{domain}/train{i}.jpg'; train.append([name] + ['0'] * 40)
            manifest[name] = {'pixel_sha256': f'{domain}_train_{i}'}
        for i in range(20):
            name = f'{domain}/val{i}.jpg'; val.append([name] + ['1'] * 40)
            pixel = f'{domain}_val_{i}'
            if i == 0: pixel = f'{domain}_train_0'
            if i == 2: pixel = f'{domain}_val_1'
            manifest[name] = {'pixel_sha256': pixel}
    (root / 'reports/data_mapped/image_manifest.json').write_text(json.dumps(manifest))
    for split, rows in [('train', train), ('val', val)]:
        path = data / f'annotations/task1/{split}/gt.csv'; path.parent.mkdir(parents=True)
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['# image'] + ATTRIBUTES); writer.writerows(rows)
    # Target duplicates with its excluded training set do not contaminate source.
    receipt = prepare(root, data, output, held)
    assert receipt['partitions']['source_train']['rows'] == 8
    assert receipt['excluded_source_val_train_pixel_duplicates'] == 2
    assert receipt['cross_partition_decoded_pixel_overlaps'] == 0
    assert set(receipt['partitions']['source_train']['domains']) == {'PA100k', 'PETA', 'Market1501'} - {held}
    assert receipt['partitions']['sealed_target_evaluation']['rows'] == 20
    before = {p.name: p.read_bytes() for p in output.glob('*.csv')}
    prepare(root, data, output, held, reuse=True)
    assert before == {p.name: p.read_bytes() for p in output.glob('*.csv')}


def test_source_lock_rejects_changed_thresholds(tmp_path):
    path = tmp_path / 'calibration.json'; path.write_text('{"threshold":0.5}')
    lock = {'files': lock_files([path])}; verify_lock(lock)
    path.write_text('{"threshold":0.4}')
    with pytest.raises(ValueError, match='changed'):
        verify_lock(lock)


def test_convergence_does_not_call_last_epoch_best_converged():
    rows = [{'metrics': {'macro': {'Challenge_Avg': .7 + i / 100}}, 'train_loss': 1 / (i + 1)} for i in range(6)]
    result = convergence(rows)
    assert result['best_at_final_epoch']
    assert not result['source_selection_plateau_heuristic']


def test_consolidation_requires_all_three_and_equal_domain_weight():
    rows = []
    for held, value in zip(['PA100k', 'PETA', 'Market1501'], [.6, .8, .7]):
        score = {'legacy': {'macro': {'Challenge_Avg': value}},
                 'mean_image_f1_sensitivity': {'macro': {'Challenge_Avg': value - .01}}}
        rows.append({'held_out_domain': held, 'target_raw': score, 'target_calibrated': score})
    assert aggregate(rows)['target_raw']['macro'] == pytest.approx(.7)
    assert aggregate(rows)['target_raw']['worst'] == .6
    with pytest.raises(ValueError, match='all three'):
        aggregate(rows[:2])
