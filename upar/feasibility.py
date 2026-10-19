"""Measure pretrained training throughput on actual official images before full training."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import UPARDataset,label_weights
from .model import UPARModel,VisionConfig
from .train import loss_weighted,build_optimizer
from .parity import code_digest


def benchmark(train_csv,data_root,parity_report,output,batch_size=16,batches=6,workers=4):
    if not torch.cuda.is_available():raise RuntimeError('GPU benchmark required; refusing CPU training')
    evidence=json.loads(Path(parity_report).read_text())
    if not evidence['passed'] or not evidence['mixed_precision_checked'] or evidence['code_sha256']!=code_digest():
        raise ValueError('GPU parity gate must pass before training benchmark')
    ck=torch.load(Path(parity_report).parent/'foundation_native.pt',map_location='cpu',weights_only=False)
    model=UPARModel(VisionConfig(**ck['vision_config']))
    model.load_state_dict(ck['state_dict']);model.freeze_vision_except_last(4);model.cuda().train()
    optimizer=build_optimizer(model,1e-5,1e-4)
    scaler=torch.amp.GradScaler('cuda')
    ds=UPARDataset(train_csv,data_root,ck['max_patches'],augment=True)
    wp,wn,_=label_weights(ds);wp=wp.cuda();wn=wn.cuda()
    # Spread benchmark samples across domains without tuning architecture.
    from collections import defaultdict
    by_domain=defaultdict(list)
    for i,row in enumerate(ds.rows):by_domain[row[3]].append(i)
    indices=[]
    for k in range(batch_size*(batches+1)):
        domain=sorted(by_domain)[k%len(by_domain)]
        indices.append(by_domain[domain][(k//len(by_domain))%len(by_domain[domain])])
    loader=DataLoader(torch.utils.data.Subset(ds,indices),batch_size=batch_size,num_workers=workers,pin_memory=True)
    times=[];losses=[];torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter();last=begin
    for index,(p,m,s,y,v,_) in enumerate(loader):
        p=p.cuda(non_blocking=True);m=m.cuda(non_blocking=True);y=y.cuda(non_blocking=True);v=v.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=loss_weighted(model(p,m,s),y,v,wp,wn)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite benchmark gradients')
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        now=time.perf_counter()
        if index:times.append(now-last);losses.append(loss.item())
        last=now
    images_per_s=batch_size/(sum(times)/len(times))
    result={'passed':True,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'batch_size':batch_size,'measured_batches':len(times),'losses':losses,
            'step_seconds':times,'images_per_second':images_per_s,
            'epoch_train_seconds_estimate':len(ds)/images_per_s,
            'estimate_limitations':'Small-batch training estimate excludes full validation, EMA comparison and checkpoint I/O; not A10 inference evidence.',
            'peak_allocated_gb':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gb':torch.cuda.max_memory_reserved()/1024**3,
            'code_sha256':code_digest(),
            'parity_report_sha256':hashlib.sha256(Path(parity_report).read_bytes()).hexdigest()}
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv',required=True);ap.add_argument('--data-root',required=True)
    ap.add_argument('--parity-report',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--batches',type=int,default=6)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    benchmark(**vars(args))

if __name__=='__main__':main()
