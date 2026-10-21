"""Candidate B limited fine-tuning workers, source-only selection/calibration."""
import argparse,gc,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset,DataLoader
from dino.v3_api import DinoV3Backbone,AttributeHead,preprocess,pool_descriptors

BASE=Path('/kaggle/working');SUITE=BASE/'dinov3_finetune';WEIGHTS=BASE/'dinov3_weights'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def load_backbone():
    from safetensors.torch import load_file
    model=DinoV3Backbone();model.load_state_dict(load_file(str(WEIGHTS/'model.safetensors')),strict=True)
    model.eval().requires_grad_(False)
    return model

class Images(Dataset):
    def __init__(self,paths,start=0,stop=None,augment=False):self.paths=paths;self.start=start;self.stop=stop if stop is not None else len(paths);self.augment=augment
    def __len__(self):return self.stop-self.start
    def __getitem__(self,i):
        index=self.start+i
        with Image.open(SUITE/'staged_data'/self.paths[index]) as image:
            image=image.convert('RGB')
            if self.augment and random.random()<.5:image=ImageOps.mirror(image)
            p,box=preprocess(image)
        return p,box,index


HEAD_HASHES={'PA100k':'97a0a0e54d2de51148b29068e3c9936ffbb45c5c944e0489867f978d8d7a317d','PETA':'1d8703ade7f3b563bd4e2b36ad95db6909c8ba4a18d2879548be8081ca276784','Market1501':'40b9bafc7888bcee37592566cd0531a10a58f9f0620ed915f5964f0931804067'}

def initial_model(held):
    from dino.limited_model import LimitedDINO
    path=SUITE/held/'INIT_HEAD.pt'
    if digest(path)!=HEAD_HASHES[held]:raise ValueError('Matching source-only frozen head bytes changed')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['config']['held_out']!=held:raise ValueError('Cross-fold or all-domain head prohibited')
    for name,sha in ck['config']['split_sha256'].items():
        if digest(SUITE/held/'splits'/name)!=sha:raise ValueError('Frozen source head partitions changed')
    head=AttributeHead();head.load_state_dict(ck['state_dict'],strict=True)
    return LimitedDINO(load_backbone(),head)

def probe(held):
    import psutil,subprocess
    from upar.data import load_rows
    from upar.train import loss_weighted
    torch.set_num_threads(2);torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
    if torch.cuda.device_count()!=1:raise RuntimeError('Probe affinity requires one visible GPU')
    rows=load_rows(SUITE/held/'splits/source_train.csv');paths=[x[0] for x in rows]
    if held in {x[3] for x in rows}:raise ValueError('Target in source benchmark')
    groups={d:[i for i,r in enumerate(rows) if r[3]==d] for d in sorted({r[3] for r in rows})};domains=sorted(groups)
    ids=[groups[domains[i%2]][(i//2)%len(groups[domains[i%2]])] for i in range(16*51)]
    # Images uses the audit index order, so benchmark indices are mapped explicitly.
    index=json.loads((SUITE/'INDEX.json').read_text());lookup={p:i for i,p in enumerate(index['paths'])}
    global_ids=[lookup[rows[i][0]] for i in ids];labels=np.load(SUITE/'labels.npz')
    loader=DataLoader(torch.utils.data.Subset(Images(index['paths'],augment=True),global_ids),batch_size=16,num_workers=2,pin_memory=True,timeout=30)
    y=torch.from_numpy(labels['y']).cuda();v=torch.from_numpy(labels['valid']).cuda()
    sy=np.stack([r[1] for r in rows]);sv=np.stack([r[2] for r in rows]);prev=np.divide(((sy==1)&sv).sum(0),sv.sum(0),out=np.full(40,.5),where=sv.sum(0)>0)
    wp=torch.tensor(np.exp(1-prev).astype(np.float32),device='cuda');wn=torch.tensor(np.exp(prev).astype(np.float32),device='cuda')
    model=initial_model(held).cuda().train();opt=model.optimizer();scaler=torch.amp.GradScaler('cuda');times=[];losses=[];waits=[]
    torch.cuda.reset_peak_memory_stats();last=time.perf_counter();grads_checked=False
    for j,(pixels,boxes,indices) in enumerate(loader):
        waits.append(time.perf_counter()-last);opt.zero_grad(set_to_none=True);ix=indices.cuda()
        with torch.autocast('cuda',dtype=torch.float16):loss=loss_weighted(model(pixels.cuda(),boxes.cuda()),y[ix],v[ix],wp,wn)
        if not torch.isfinite(loss):raise FloatingPointError('Probe loss nonfinite')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        for name,p in model.named_parameters():
            if not p.requires_grad and p.grad is not None:raise ValueError('Frozen parameter received gradient '+name)
            if p.requires_grad and p.grad is None:raise ValueError('Trainable parameter missing gradient '+name)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):
            scaler.step(opt);scaler.update();last=time.perf_counter();continue
        grads_checked=True;scaler.step(opt);scaler.update();torch.cuda.synchronize();now=time.perf_counter()
        if j>=3:times.append(now-last);losses.append(float(loss))
        last=now
    if len(times)<40 or not grads_checked:raise ValueError('Insufficient finite backward measurements')
    result={'source_only':True,'held_out':held,'passed':True,'measured_batches':len(times),'images_per_second':16*len(times)/sum(times),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'loader_wait_seconds':sum(waits),'losses':losses,'gradient_freeze_verified':True,'trainable_backbone':['blocks.10','blocks.11','norm'],'cpu_percent':psutil.cpu_percent(interval=.2),'rss_bytes':psutil.Process().memory_info().rss,'gpu_status':subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}
    if result['peak_allocated_GiB']>12 or result['images_per_second']<60:raise RuntimeError('Backward resource gate below60ips or above12GiB')
    write(SUITE/(held+'_PROBE.json'),result)

def head(held):
    from upar.data import load_rows
    from upar.attributes import ATTRIBUTES
    from upar.train import loss_weighted
    from upar.metrics import macro_score
    from upar.checkpoint import atomic_save,rng_state,restore_rng
    from scripts.generalization_analysis import fit_fast,sensitivity,attribute_rows
    from upar.lodo import group_report
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
    if torch.cuda.device_count()!=1:raise RuntimeError('Head requires independent one-GPU affinity')
    fold=SUITE/held;fold.mkdir(exist_ok=True);begin=time.monotonic()
    if (fold/'LOCK.json').exists():
        lock=json.loads((fold/'LOCK.json').read_text())
        for name,sha in lock['files'].items():
            if digest(SUITE/name)!=sha:raise ValueError('Saved pre-target lock changed')
        if (fold/'EVALUATION.json').exists():
            result=json.loads((fold/'EVALUATION.json').read_text())
            if digest(fold/'target_predictions.npz')!=result['target_prediction_sha256']:raise ValueError('Saved target predictions changed')
            print(held,'Reusing completed locked result; no target reevaluation',flush=True);return
        raise RuntimeError('Preserved pre-target lock requires recovery without refitting or repeated inference')
    index=json.loads((SUITE/'INDEX.json').read_text());paths=index['paths'];lookup={p:i for i,p in enumerate(paths)}
    labels=np.load(SUITE/'labels.npz');y=torch.from_numpy(labels['y']).cuda();v=torch.from_numpy(labels['valid']).cuda()
    split=fold/'splits';parts={n:load_rows(split/(n+'.csv')) for n in ['source_train','source_selection','source_calibration','sealed_target_evaluation']}
    for name in ['source_train','source_selection','source_calibration']:
        if held in {r[3] for r in parts[name]}:raise ValueError('Target present in source partition')
    ix={n:np.array([lookup[r[0]] for r in rows],dtype=np.int64) for n,rows in parts.items()}
    for name,rows in parts.items():
        if not np.array_equal(labels['y'][ix[name]],np.stack([r[1] for r in rows])) or not np.array_equal(labels['valid'][ix[name]],np.stack([r[2] for r in rows])):raise ValueError('Cache-label alignment differs')
    source=parts['source_train'];sy=np.stack([r[1] for r in source]);sv=np.stack([r[2] for r in source])
    prevalence=np.divide(((sy==1)&sv).sum(0),sv.sum(0),out=np.full(40,.5),where=sv.sum(0)>0)
    wp=torch.tensor(np.exp(1-prevalence).astype(np.float32),device='cuda');wn=torch.tensor(np.exp(prevalence).astype(np.float32),device='cuda')
    from collections import Counter
    counts=Counter(r[3] for r in source)
    weights=torch.tensor([counts[r[3]]**-.5 for r in source],dtype=torch.double)
    generator=torch.Generator().manual_seed(2027)
    config={'method':'candidate_B_final_two_blocks','attributes':ATTRIBUTES,'held_out':held,'seed':2027,'frozen_backbone':False,'trainable_backbone':['blocks.10','blocks.11','norm'],'frozen_head_initialization_sha256':digest(fold/'INIT_HEAD.pt'),'image_augmentation':'horizontal flip probability0.5 only','sampler_alpha':.5,'head_lr':5e-5,'backbone_lr':5e-6,'dropout':.2,'microbatch':16,'accumulate':4,'max_epochs':6,'fixed_learning_rates':True,'weight_decay':.01,'ensemble_siglip_weight':.75,'canvas':[320,160],'backbone':'timm DINOv3 ViT-B/16 FP32 RoPE','foundation_sha256':digest(WEIGHTS/'model.safetensors'),'split_sha256':{p.name:digest(p) for p in split.glob('*.csv')}}
    write(fold/'CONFIG.json',config)
    write(fold/'SOURCE_SUPERVISION.json',{'domains':dict(counts),'positive_counts':((sy==1)&sv).sum(0).tolist(),'negative_counts':((sy==0)&sv).sum(0).tolist(),'prevalence':prevalence.tolist(),'positive_weights':wp.cpu().tolist(),'negative_weights':wn.cpu().tolist(),'target_excluded':True})
    model=initial_model(held).cuda();optimizer=model.optimizer();scaler=torch.amp.GradScaler('cuda')
    def predict(indices):
        model.eval();out=[]
        loader=DataLoader(torch.utils.data.Subset(Images(paths),indices.tolist()),batch_size=64,num_workers=2,pin_memory=True,timeout=30)
        with torch.inference_mode():
            for p,boxes,_ in loader:
                with torch.autocast('cuda',dtype=torch.float16):logits=model(p.cuda(non_blocking=True),boxes.cuda(non_blocking=True))
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out)
    best=-1.;bad=0;history=[];start=0;seen=np.zeros(len(source),dtype=bool)
    if (fold/'last.pt').exists():
        ck=torch.load(fold/'last.pt',map_location='cpu',weights_only=False)
        if ck['config']!=config:raise ValueError('Resume config mismatch')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'],generator)
        start=ck['epoch']+1;best=ck['best'];bad=ck['bad'];history=ck['history'];seen=ck['seen_indices']
    for epoch in range(start,6):
        tick=time.monotonic();model.train();chosen=torch.multinomial(weights,len(source),replacement=True,generator=generator).numpy()
        seen[chosen]=True
        unique_epoch=Counter(source[j][3] for j in np.unique(chosen));unique_cumulative=Counter(source[j][3] for j in np.flatnonzero(seen))
        draws=Counter(source[j][3] for j in chosen);total_loss=0.;updates=0;skipped=0
        loader=DataLoader(torch.utils.data.Subset(Images(paths,augment=True),ix['source_train'][chosen].tolist()),batch_size=16,num_workers=2,pin_memory=True,timeout=30)
        optimizer.zero_grad(set_to_none=True);batches=len(loader)
        for j,(pixels,boxes,indices) in enumerate(loader):
            batch=indices.cuda(non_blocking=True);group_start=(j//4)*4;divisor=min(4,batches-group_start)
            with torch.autocast('cuda',dtype=torch.float16):loss=loss_weighted(model(pixels.cuda(non_blocking=True),boxes.cuda(non_blocking=True)),y[batch],v[batch],wp,wn)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite limited-fine-tuning loss')
            scaler.scale(loss/divisor).backward();total_loss+=loss.item()
            if (j+1)%4==0 or j+1==batches:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                previous=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped+=int(scaler.get_scale()<previous);updates+=1;optimizer.zero_grad(set_to_none=True)
            if j%32==0:
                elapsed=time.monotonic()-tick
                write(fold/'PROGRESS.json',{'epoch':epoch+1,'batch':j+1,'batches':batches,'epoch_elapsed_seconds':elapsed,'training_epoch_eta_seconds':elapsed/max(1,j+1)*(batches-j-1),'elapsed_seconds':time.monotonic()-begin,'complete':False})
        write(fold/'PROGRESS.json',{'epoch':epoch+1,'phase':'source_selection','elapsed_seconds':time.monotonic()-begin,'complete':False})
        prob=predict(ix['source_selection']);rows=parts['source_selection'];score=macro_score(prob,np.stack([r[1] for r in rows]),np.stack([r[2] for r in rows]),[r[3] for r in rows])['macro']['Challenge_Avg']
        row={'epoch':epoch+1,'loss':total_loss/max(1,batches),'source_selection_score':score,'draws':dict(draws),'unique_epoch':dict(unique_epoch),'unique_cumulative':dict(unique_cumulative),'updates':updates,'amp_skips':skipped,'epoch_seconds':time.monotonic()-tick};history.append(row)
        old_best=best
        if score>best:
            best=score;atomic_save({'state_dict':model.state_dict(),'epoch':epoch+1,'source_selection_score':score,'config':config},fold/'best.pt')
        bad=0 if score>old_best+1e-4 else bad+1
        atomic_save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(generator),'epoch':epoch,'best':best,'bad':bad,'history':history,'seen_indices':seen,'config':config},fold/'last.pt')
        write(fold/'HISTORY.json',history);write(fold/'PROGRESS.json',{'epoch':epoch+1,'source_selection_score':score,'elapsed_seconds':time.monotonic()-begin,'complete':False});print(held,json.dumps(row),flush=True)
        # All six predeclared epochs, source-only checkpoint selection.
    ck=torch.load(fold/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'])
    selection_rows=parts['source_selection'];selection_prob=predict(ix['source_selection'])
    np.savez_compressed(fold/'source_selection_predictions.npz',prob=selection_prob,y=np.stack([r[1] for r in selection_rows]),valid=np.stack([r[2] for r in selection_rows]),domains=np.array([r[3] for r in selection_rows]),image_paths=np.array([r[0] for r in selection_rows]))
    cal_rows=parts['source_calibration'];ci=ix['source_calibration'];pcal=predict(ci)
    cy=np.stack([r[1] for r in cal_rows]);cv=np.stack([r[2] for r in cal_rows]);cd=np.array([r[3] for r in cal_rows])
    oldcal=np.load(SUITE/'siglip'/held/'source_calibration_predictions.npz')
    if not np.array_equal(oldcal['y'],cy) or not np.array_equal(oldcal['valid'],cv) or not np.array_equal(oldcal['domains'],cd):raise ValueError('SigLIP source calibration alignment differs')
    mixedcal=.75*oldcal['prob']+.25*pcal
    threshold=fit_fast(pcal,cy,cv,cd);mixed_threshold=fit_fast(mixedcal,cy,cv,cd)
    write(fold/'CALIBRATION.json',{'dino_thresholds':threshold.tolist(),'ensemble_thresholds':mixed_threshold.tolist(),'source_domains':sorted(set(cd)),'ensemble_weight_siglip':.75,'prediction_interpretation':'weighted raw sigmoid probabilities, then one source-only threshold vector','target_used':False})
    np.savez_compressed(fold/'source_calibration_predictions.npz',prob=pcal,y=cy,valid=cv,domains=cd,image_paths=np.array([r[0] for r in cal_rows]))
    # Checkpoint and both calibrations fixed before target inference or target metrics.
    lock={str(p.relative_to(SUITE)):digest(p) for p in [fold/'best.pt',fold/'CONFIG.json',fold/'CALIBRATION.json',fold/'source_calibration_predictions.npz',fold/'source_selection_predictions.npz',*split.glob('*.csv')]}
    write(fold/'LOCK.json',{'files':lock,'source_only_selection_and_calibration':True,'target_evaluation_count':1})
    tr=parts['sealed_target_evaluation'];ty=np.stack([r[1] for r in tr]);tv=np.stack([r[2] for r in tr]);td=np.array([r[3] for r in tr]);p=predict(ix['sealed_target_evaluation'])
    old=np.load(SUITE/'siglip'/held/'target_predictions.npz')
    if not np.array_equal(old['y'],ty) or not np.array_equal(old['valid'],tv) or not np.array_equal(old['domains'],td):raise ValueError('SigLIP target alignment differs')
    mix=.75*old['prob']+.25*p
    np.savez_compressed(fold/'target_predictions.npz',prob=p,ensemble_prob=mix,y=ty,valid=tv,domains=td,image_paths=np.array([r[0] for r in tr]))
    reports={};attributes=[];groups={}
    for name,prob,t in [('dino_raw',p,np.full(40,.5)),('dino_calibrated',p,threshold),('ensemble_raw',mix,np.full(40,.5)),('ensemble_calibrated',mix,mixed_threshold)]:
        reports[name]=sensitivity(prob,ty,tv,td,t);a=attribute_rows(prob,ty,tv,td,t,name);attributes+=a;groups[name]=group_report(a)
    mask=tv.astype(bool);e1=(old['prob']-ty)[mask];e2=(p-ty)[mask]
    corr=float(np.corrcoef(e1,e2)[0,1]);binary1=(old['prob']>.5)!=ty.astype(bool);binary2=(p>.5)!=ty.astype(bool)
    error={'raw_probability_residual_correlation':corr,'binary_error_correlation_at_half':float(np.corrcoef(binary1[mask],binary2[mask])[0,1]),'raw_decision_disagreement':float(((old['prob']>.5)!=(p>.5))[mask].mean()),'double_fault_at_half':float((binary1&binary2)[mask].mean())}
    for path,expected in lock.items():
        if digest(SUITE/path)!=expected:raise ValueError('Source lock changed during target evaluation')
    write(fold/'ATTRIBUTE_METRICS.json',attributes)
    result={'held_out':held,'reports':reports,'attribute_groups':groups,'error_complementarity':error,'epochs':len(history),'best_epoch':ck['epoch'],'source_selection_score':ck['source_selection_score'],'elapsed_seconds':time.monotonic()-begin,'source_train_rows':len(source),'lock_sha256':digest(fold/'LOCK.json'),'target_prediction_sha256':digest(fold/'target_predictions.npz'),'target_rows':len(tr),'target_evaluations':1}
    write(fold/'EVALUATION.json',result);write(fold/'PROGRESS.json',{'complete':True,'epochs':len(history),'elapsed_seconds':time.monotonic()-begin})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--probe');a.add_argument('--head');p=a.parse_args()
    if p.probe:probe(p.probe)
    elif p.head:head(p.head)
