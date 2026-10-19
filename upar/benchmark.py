"""End-to-end validator for offline submission. Benchmarks decode+preprocess+infer."""
from __future__ import annotations
import argparse,json,sys,time
PROCESS_START=time.perf_counter()
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--submission-dir',required=True)
    ap.add_argument('--images-dir')
    ap.add_argument('--samples-json',help='Explicit image paths; avoids inventory scans absent from the official API')
    ap.add_argument('--n',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--assumed-hidden-n',type=int,default=33407)
    ap.add_argument('--output')
    args=ap.parse_args()
    sys.path.insert(0,str(Path(args.submission_dir).resolve()))
    import run
    from upar.attributes import ATTRIBUTES
    from PIL import Image
    discovery_start=time.perf_counter()
    if args.samples_json:
        images=[Path(name) for name in json.loads(Path(args.samples_json).read_text())]
    elif args.images_dir:
        images=list(Path(args.images_dir).rglob('*.jpg'))+list(Path(args.images_dir).rglob('*.png'))
    else:
        ap.error('Provide --samples-json or --images-dir')
    if not images:raise FileNotFoundError('No images available for benchmark')
    images=(images*(args.n//len(images)+1))[:args.n]
    discovery_sec=time.perf_counter()-discovery_start
    sample=lambda i,path:{'index':i,'image_path':str(path.resolve()),
                          'image':path.name,'attribute_names':ATTRIBUTES}
    start=time.perf_counter();run.load_model()
    if torch.cuda.is_available():torch.cuda.synchronize()
    load_sec=time.perf_counter()-start
    # Warmup is part of cold-start overhead, and is included in projection.
    warm_start=time.perf_counter()
    run.predict_batch([sample(k,p) for k,p in enumerate(images[:min(args.batch_size,len(images))])])
    if torch.cuda.is_available():torch.cuda.synchronize()
    warm_sec=time.perf_counter()-warm_start
    start=time.perf_counter()
    total=0
    for k in range(0,len(images),args.batch_size):
        batch=[sample(k+q,p) for q,p in enumerate(images[k:k+args.batch_size])]
        preds=run.predict_batch(batch)
        if len(preds)!=len(batch): raise AssertionError('Wrong result count')
        import numpy as np
        array=np.asarray(preds)
        if array.shape!=(len(batch),40) or not np.isfinite(array).all() or not ((array>=0)&(array<=1)).all():
            raise AssertionError('Invalid probabilities')
        json.dumps(preds,allow_nan=False)
        total+=len(batch)
    if torch.cuda.is_available():torch.cuda.synchronize()
    sec=time.perf_counter()-start
    ips=total/sec
    measured_total=time.perf_counter()-PROCESS_START
    overhead=measured_total-sec
    est=overhead+args.assumed_hidden_n/ips
    result={'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'load_s':load_sec,'warmup_s':warm_sec,'images':total,'elapsed_s':sec,'images_per_s':ips,
            'measured_total_s':measured_total,'serialization_included':True,
            'sample_discovery_s':discovery_sec,'explicit_sample_paths':bool(args.samples_json),
            'submission_api_total_s':load_sec+warm_sec+sec,
            'hypothetical_hidden_images':args.assumed_hidden_n,
            'projected_runtime_s':est,'hypothetical_under_300s':est<300,
            'limitation':'Hypothetical workload only. Hidden workload unknown. Hardware differs from official A10 unless explicitly matched. This does not certify the official 300s limit.'}
    print(json.dumps(result,indent=2))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
