"""Source-only real throughput gate; no production or target model is opened."""
import argparse,json,os,random,sys,time
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--held',required=True);ap.add_argument('--output',required=True);ap.add_argument('--barrier');a=ap.parse_args()
    code=Path('/kaggle/working/lodo_code');sys.path.insert(0,str(code));os.chdir(code)
    import numpy as np,torch,psutil
    torch.set_num_threads(2);torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
    if torch.cuda.device_count()!=1:raise RuntimeError('Probe requires explicit single-GPU affinity')
    from upar.feasibility import benchmark
    from upar.data import UPARDataset
    suite=code.parent/'lodo';train=suite/a.held/'splits/source_train.csv'
    ds=UPARDataset(train,suite/'staged_data',256,augment=True)
    # CPU preprocessing and loader timing from source rows only; discarded afterwards.
    from collections import defaultdict
    groups=defaultdict(list)
    for i,row in enumerate(ds.rows):groups[row[3]].append(i)
    domains=sorted(groups);indices=[groups[domains[k%len(domains)]][(k//len(domains))%len(groups[domains[k%len(domains)]])] for k in range(192)]
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=16,num_workers=2)
    start=time.monotonic();images=0
    for batch in loader:images+=len(batch[3])
    loader_s=time.monotonic()-start
    if a.barrier:
        barrier=Path(a.barrier);Path(str(barrier)+'.'+a.held+'.ready').write_text('ready')
        wait=time.monotonic()
        while not barrier.exists():
            if time.monotonic()-wait>90:raise TimeoutError('Concurrent probe barrier timed out')
            time.sleep(.1)
    probe_start=time.time()
    r=benchmark(str(train),str(suite/'staged_data'),str(suite/'gates/PRETRAINED_PARITY.json'),a.output,
        batch_size=16,batches=48,workers=2)
    r.update({'source_only':True,'held_out':a.held,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'visible_gpu_count':torch.cuda.device_count(),'loader_images_per_second':images/loader_s,
        'loader_seconds':loader_s,'cpu_logical_count':os.cpu_count(),'cpu_util_percent_end':psutil.cpu_percent(interval=.2),
        'process_rss_bytes':psutil.Process().memory_info().rss,'workers':2,'torch_threads':2,
        'probe_start_unix':probe_start,'probe_end_unix':time.time()})
    Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()
