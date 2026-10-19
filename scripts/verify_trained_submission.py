"""Check an actual trained archive with optional dependencies/network blocked."""
import argparse
import hashlib
import importlib.abc
import json
import socket
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',required=True)
    ap.add_argument('--image',action='append',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--metadata',default=str(Path(__file__).resolve().parents[1]/'submission_template/metadata.yaml'))
    args=ap.parse_args()
    torch.set_num_threads(4)
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    unpacked=output/'submission';unpacked.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        names=archive.namelist()
        expected={'run.py','metadata.yaml','assets/best.pt','assets/calibration.json',
                  'upar/__init__.py','upar/model.py','upar/preprocess.py','upar/attributes.py'}
        if len(names)!=len(set(names)) or set(names)!=expected:raise AssertionError('Unexpected archive structure')
        for name in names:
            if not (unpacked/name).resolve().is_relative_to(unpacked):raise ValueError('Unsafe archive member')
        if archive.testzip() is not None:raise AssertionError('ZIP CRC failure')
        if archive.read('metadata.yaml')!=Path(args.metadata).read_bytes():raise AssertionError('Metadata bytes changed')
        archive.extractall(unpacked)
    checkpoint=torch.load(unpacked/'assets/best.pt',map_location='cpu',weights_only=False)
    if not checkpoint.get('trained') or checkpoint.get('test_fixture'):raise AssertionError('Actual trained weights required')
    epoch=int(checkpoint['epoch'])+1
    selected=checkpoint.get('selected')
    query_shape=list(checkpoint['state_dict']['head.attribute_queries'].shape)
    if query_shape!=[40,768]:raise AssertionError('Unexpected semantic queries')
    attributes=checkpoint['attributes']
    del checkpoint
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,*unused):
            if fullname.split('.')[0] in {'transformers','huggingface_hub','timm','requests'}:
                raise ImportError('Blocked optional dependency: '+fullname)
    sys.meta_path.insert(0,Block())
    def denied(*unused,**kwargs):raise RuntimeError('Network forbidden during inference')
    socket.socket=denied;socket.create_connection=denied
    sys.path.insert(0,str(unpacked))
    import run
    start=time.perf_counter();run.load_model();load_s=time.perf_counter()-start
    samples=[{'image_path':str(Path(name).resolve()),'attribute_names':attributes} for name in args.image]
    if not all(Path(s['image_path']).is_file() for s in samples):raise FileNotFoundError('Actual images required')
    if run.predict_batch([])!=[]:raise AssertionError('Empty batch failed')
    batch=[samples[i%len(samples)] for i in range(35)]
    results=np.asarray(run.predict_batch(batch))
    if results.shape!=(35,40) or not np.isfinite(results).all() or not ((results>=0)&(results<=1)).all():
        raise AssertionError('Invalid probabilities')
    single=np.asarray(run.predict_image(samples[0]))
    delta=float(np.max(np.abs(single-results[0])))
    if delta>(.002 if run._DEVICE.type=='cuda' else 1e-6):raise AssertionError('Single/batch mismatch')
    try:run.predict_image({**samples[0],'attribute_names':attributes[::-1]})
    except ValueError:pass
    else:raise AssertionError('Incorrect attribute order accepted')
    json.dumps(results.tolist(),allow_nan=False)
    report={'passed':True,'python':sys.version,'torch':torch.__version__,'device':str(run._DEVICE),
            'archive_sha256':sha256(args.archive),'checkpoint_sha256':sha256(unpacked/'assets/best.pt'),
            'metadata_sha256':sha256(args.metadata),'epoch':epoch,'selected':selected,'query_shape':query_shape,
            'actual_image_count':len(samples),'batch_count':35,'single_batch_max_abs':delta,'load_s':load_s,
            'network_blocked':True,'optional_dependencies_blocked':True,'attributes':attributes,
            'samples':[{'image_sha256':sha256(s['image_path']),'probabilities':results[i].tolist()}
                       for i,s in enumerate(samples)]}
    (output/'TRAINED_OFFLINE_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'attributes','samples'}},indent=2))


if __name__=='__main__':main()
