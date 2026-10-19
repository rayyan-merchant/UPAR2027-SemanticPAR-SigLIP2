"""Read-only full annotation, decode and exact-duplicate audit; no torch needed."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from .attributes import ATTRIBUTES


def read_annotations(path):
    raw = Path(path).read_bytes()
    reader = csv.reader(raw.decode('utf-8-sig').splitlines())
    header = next(reader)
    if header[0].strip() != '# image' or header[1:] != ATTRIBUTES:
        raise ValueError('Official attribute order/header mismatch')
    rows, invalid = [], Counter()
    for index, row in enumerate(reader, 2):
        if len(row) != 41:
            raise ValueError(f'{path}:{index}: expected 41 columns')
        name = row[0].strip().replace('\\', '/')
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in name:
            raise ValueError(f'Unsafe image path {name}')
        values = np.array([float(v) if v.strip() else np.nan for v in row[1:]], np.float32)
        valid = np.isfinite(values) & ((values == 0) | (values == 1))
        invalid.update(v for v, ok in zip(row[1:], valid) if not ok)
        rows.append((name, values, valid, p.parts[0]))
    return rows, {'sha256': hashlib.sha256(raw).hexdigest(), 'encoding': 'UTF-8 (BOM accepted)',
                  'invalid_tokens': dict(invalid)}


def audit(train_csv, val_csv, root, output, workers=12, peta_mapping=None):
    start = time.perf_counter()
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, ts = read_annotations(train_csv)
    val, vs = read_annotations(val_csv)
    paths = sorted({r[0] for r in train + val})
    aliases = {}
    if peta_mapping:
        with open(peta_mapping, newline='', encoding='utf-8-sig') as mapping_file:
            for source, destination in csv.reader(mapping_file):
                if destination in aliases:
                    raise ValueError(f'Duplicate official mapping: {destination}')
                aliases[destination] = source
    resolutions = {}
    def inspect(name):
        path = root / name
        if not path.is_file() and name in aliases:
            path = root / aliases[name]
        try:
            if not path.is_file():
                return name, {'error': 'missing'}
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:
                im.load()
                rgb = im.convert('RGB')
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                return name, {'source_relative_path':path.relative_to(root).as_posix(),
                              'sha256': file_hash, 'pixel_sha256': pixels,
                              'size': list(rgb.size), 'mode': im.mode}
        except Exception as exc:
            return name, {'error': f'{type(exc).__name__}: {exc}'}
    evidence = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (name, result) in enumerate(pool.map(inspect, paths), 1):
            evidence[name] = result
            if index % 10000 == 0:
                print(f'Decoded/hashed {index}/{len(paths)}', flush=True)
    stats = {}
    for split, rows, meta in [('train', train, ts), ('val', val, vs)]:
        counts, domains = Counter(r[3] for r in rows), {}
        for domain, n in sorted(counts.items()):
            selected = [r for r in rows if r[3] == domain]
            y, v = np.stack([r[1] for r in selected]), np.stack([r[2] for r in selected])
            pos, neg = ((y == 1) & v).sum(0), ((y == 0) & v).sum(0)
            prevalence = np.divide(pos, pos + neg, out=np.full(40, np.nan), where=pos+neg > 0)
            domains[domain] = {'records': n, 'positive': pos.tolist(), 'negative': neg.tolist(),
                               'invalid': (~v).sum(0).tolist(),
                               'prevalence': [None if np.isnan(x) else float(x) for x in prevalence]}
        stats[split] = {**meta, 'records': len(rows), 'domains': domains,
                        'duplicate_paths_within_split': len(rows)-len({r[0] for r in rows})}
    train_paths, val_paths = {r[0] for r in train}, {r[0] for r in val}
    duplicates = {}
    for key in ['sha256', 'pixel_sha256']:
        by_hash = defaultdict(list)
        for name, item in evidence.items():
            if key in item:
                by_hash[item[key]].append(name)
        duplicates[key] = [names for names in by_hash.values()
                           if any(n in train_paths for n in names) and any(n in val_paths for n in names)]
    invalid_files = {n: e for n, e in evidence.items() if 'error' in e}
    frequency = Counter(r[3] for r in train)
    mass = {d: n**.5 for d, n in frequency.items()}
    mass = {d: value/sum(mass.values()) for d, value in mass.items()}
    manifest_path = output/'image_manifest.json'
    manifest_path.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    result = {'schema': 1, 'data_root': str(root), 'attributes': ATTRIBUTES, 'splits': stats,
              'official_mapping':str(peta_mapping) if peta_mapping else None,
              'official_mapping_sha256':hashlib.sha256(Path(peta_mapping).read_bytes()).hexdigest() if peta_mapping else None,
              'mapped_paths':sum(e.get('source_relative_path',n)!=n for n,e in evidence.items()),
              'unique_annotated_paths': len(paths), 'fully_decoded': len(paths)-len(invalid_files),
              'file_errors': invalid_files, 'path_overlap': sorted(train_paths & val_paths),
              'cross_split_duplicates': duplicates, 'sampling_domain_mass': mass,
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'elapsed_seconds': time.perf_counter()-start,
              'passed': not invalid_files and not (train_paths & val_paths),
              'leakage_review_required': bool(duplicates['pixel_sha256']),
              'limitations': 'Exact byte/pixel duplicates only; near duplicates and shared identities need separate review.'}
    (output/'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Data audit', '', 'Read-only audit of actual official annotations and all referenced images.', '',
             f'Train: {len(train):,}; validation: {len(val):,}; unique paths: {len(paths):,}.',
             f'Fully decoded: {result["fully_decoded"]:,}; file errors: {len(invalid_files)}.',
             f'Path overlap: {len(result["path_overlap"])}; cross-split byte duplicate groups: {len(duplicates["sha256"])};',
             f'cross-split decoded-pixel duplicate groups: {len(duplicates["pixel_sha256"])}.', '',
             '40 ordered attribute names match both official CSV headers.',
             f"Paths resolved via official PETA mapping: {result['mapped_paths']:,}. Source files unchanged.", '',
             '| Split | Domain | Records | Invalid label cells |', '|---|---|---:|---:|']
    for split in ['train','val']:
        for domain, group in stats[split]['domains'].items():
            lines.append(f'| {split} | {domain} | {group["records"]:,} | {sum(group["invalid"]):,} |')
    lines += ['', 'Expected domain sampling mass (per-image N_domain^-0.5):', '',
              *[f'- {d}: {p:.6f}' for d,p in mass.items()], '',
              'Per-domain, per-attribute positive/negative counts and prevalence: data_audit.json.',
              'Image dimensions, byte and decoded-pixel hashes: image_manifest.json.', '', result['limitations'],
              'Duplicate findings require review; original splits have not been changed.']
    (output/'DATA_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['fully_decoded','unique_annotated_paths','passed','leakage_review_required','elapsed_seconds']}), flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-csv',required=True);p.add_argument('--val-csv',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--output',default='reports/data')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--peta-mapping')
    args=p.parse_args()
    result=audit(args.train_csv,args.val_csv,args.data_root,args.output,args.workers,args.peta_mapping)
    if not result['passed']:
        raise SystemExit('Data integrity gate failed; inspect report')

if __name__=='__main__':main()
