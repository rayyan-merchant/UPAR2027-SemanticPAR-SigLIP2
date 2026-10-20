"""Official UPAR gt.csv loading, invalid labels and domain-balanced sampling."""
from __future__ import annotations
import csv
import os
from pathlib import Path, PurePosixPath
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from .attributes import ATTRIBUTES
from .preprocess import preprocess_image


def load_rows(csv_path):
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header and header[0].strip().lower() in ('# image','image','path','filename'):
            names = [x.strip() for x in header[1:]]
        else:
            raise ValueError(f"Unexpected first field in CSV header: {header[:3]}")
        if names != ATTRIBUTES:
            raise ValueError('Official CSV attributes differ from the expected ordered list. Inspect dataset version.')
        rows=[]
        for line in reader:
            if not line or len(line) != 41:
                if not line: continue
                raise ValueError(f'Expected 41 columns, got {len(line)} on row {len(rows)+2}')
            path=line[0].strip()
            path=path.replace('\\', '/')
            parsed=PurePosixPath(path)
            if parsed.is_absolute() or '..' in parsed.parts or ':' in path:
                raise ValueError(f'Unsafe annotation path: {path}')
            labels=np.asarray([float(z) if z.strip() else np.nan for z in line[1:]],dtype=np.float32)
            valid=np.isfinite(labels) & np.logical_or(labels == 0,labels == 1)
            domain=path.replace('\\','/').split('/')[0]
            rows.append((path, labels, valid, domain))
    return rows


class UPARDataset(Dataset):
    def __init__(self, annotations, data_root, max_patches=256, augment=False, check_files=False):
        self.rows=load_rows(annotations)
        self.root=Path(data_root)
        self.max_patches=max_patches
        self.augment=augment
        if check_files:
            missing=[path for path,*_ in self.rows if not (self.root/path).is_file()]
            if missing:
                raise FileNotFoundError(f'{len(missing)} dataset images missing, e.g. {missing[:4]}')

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        path, label,valid,domain=self.rows[i]
        with Image.open(self.root / path) as im:
            pixels,patch_mask,shapes=preprocess_image(im,self.max_patches,augment=self.augment)
        return (pixels,patch_mask,shapes,
                torch.from_numpy(label.copy()),torch.from_numpy(valid.copy()),domain)


class ExposureSampler(WeightedRandomSampler):
    """Observe exact sampler draws without changing its RNG stream.

    Only complete epochs are committed. Index sets are private resume state;
    public telemetry contains aggregate counts, never image paths/identities.
    """
    def __init__(self, weights, domains, generator):
        super().__init__(weights, num_samples=len(domains), replacement=True, generator=generator)
        self.domains = list(domains)
        self.cumulative = {d: set() for d in sorted(set(domains))}
        self.counts = {}; self.unique = {}

    def __iter__(self):
        self.counts = {d: 0 for d in self.cumulative}
        self.unique = {d: set() for d in self.cumulative}
        for index in super().__iter__():
            domain = self.domains[index]
            self.counts[domain] += 1; self.unique[domain].add(index)
            self.cumulative[domain].add(index)
            yield index

    def exposure(self):
        total = sum(self.counts.values())
        return {'draws': dict(self.counts),
                'fractions': {d: n / max(1, total) for d, n in self.counts.items()},
                'unique_images_epoch': {d: len(v) for d, v in self.unique.items()},
                'unique_images_cumulative': {d: len(v) for d, v in self.cumulative.items()}}

    def exposure_state(self):
        return {d: sorted(v) for d, v in self.cumulative.items()}

    def restore_exposure(self, state):
        if set(state) != set(self.cumulative): raise ValueError('Exposure domains changed')
        for d, values in state.items():
            if any(not isinstance(i, int) or not 0 <= i < len(self.domains) or self.domains[i] != d for i in values):
                raise ValueError('Invalid exposure resume indices')
        self.cumulative = {d: set(v) for d, v in state.items()}


def domain_sampler(dataset, alpha=0.5, generator=None, track_exposure=False):
    from collections import Counter
    freq=Counter(x[3] for x in dataset.rows)
    weights=[freq[r[3]]**(alpha-1) for r in dataset.rows]
    if track_exposure:
        return ExposureSampler(weights, [r[3] for r in dataset.rows], generator)
    return WeightedRandomSampler(weights,num_samples=len(dataset),replacement=True,generator=generator)


def label_weights(dataset):
    y=np.stack([r[1] for r in dataset.rows],axis=0)
    v=np.stack([r[2] for r in dataset.rows],axis=0)
    p=np.divide(((y == 1)&v).sum(0),v.sum(0),out=np.full(40,0.5),where=v.sum(0)>0)
    wp=np.exp(1-p).astype(np.float32)
    wn=np.exp(p).astype(np.float32)
    return torch.from_numpy(wp),torch.from_numpy(wn),p
