"""Candidate B limited fine-tuning workers, source-only selection/calibration."""
import argparse,gc,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset,DataLoader
from dino.v3_api import DinoV3Backbone,AttributeHead,preprocess,pool_descriptors

BASE=Path('/kaggle/working');SUITE=BASE/'candidate_b_final';WEIGHTS=BASE/'dinov3_weights'

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


HEAD_HASHES={'all_domain':'6c2943f3cff1c0c2840e327ee6feea5aa7a1bf6d2f9ce0bf779e40c5aa5ec4f4'}

def extension_allowed(history, completed):
    if completed < 6:return True
    if completed >= 8:return False
    scores=[float(x['source_selection_score']) for x in history]
    return scores[-1] >= max(scores[:-1])+1e-4

def initial_model(held):
    from dino.limited_model import LimitedDINO
    from upar.attributes import ATTRIBUTES
    if held!='all_domain':raise ValueError('Final job is not a LODO fold')
    path=SUITE/held/'INIT_HEAD.pt'
    if digest(path)!=HEAD_HASHES[held]:raise ValueError('Candidate A selected head changed')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['config']['candidate']!='A' or not ck['config']['frozen_backbone'] or ck['config']['attributes']!=ATTRIBUTES or ck['epoch']!=30:raise ValueError('Wrong frozen all-domain head')
    h=AttributeHead();h.load_state_dict(ck['state_dict'],strict=True)
    return LimitedDINO(load_backbone(),h)


def probe(held):
    import psutil,subprocess
    from upar.data import load_rows
    from upar.train import loss_weighted
    torch.set_num_threads(2);torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
    if torch.cuda.device_count()!=1:raise RuntimeError('Probe affinity requires one visible GPU')
    rows=load_rows(SUITE/held/'splits/source_train.csv');paths=[x[0] for x in rows]
    if held in {x[3] for x in rows}:raise ValueError('Target in source benchmark')
    groups={d:[i for i,r in enumerate(rows) if r[3]==d] for d in sorted({r[3] for r in rows})};domains=sorted(groups)
    ids=[groups[domains[i%len(domains)]][(i//len(domains))%len(groups[domains[i%len(domains)]])] for i in range(16*51)]
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
            if digest(fold/name)!=sha:raise ValueError('Completed source lock changed')
        for name,sha in lock['split_files'].items():
            if digest(fold/'splits'/name)!=sha:raise ValueError('Completed split changed')
        if not (fold/'EVALUATION.json').exists():raise RuntimeError('Locked result incomplete')
        return
    index=json.loads((SUITE/'INDEX.json').read_text());paths=index['paths'];lookup={p:i for i,p in enumerate(paths)}
    labels=np.load(SUITE/'labels.npz');y=torch.from_numpy(labels['y']).cuda();v=torch.from_numpy(labels['valid']).cuda()
    split=fold/'splits';parts={n:load_rows(split/(n+'.csv')) for n in ['source_train','source_selection','source_calibration']}
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
    config={'method':'candidate_B_final_all_domain_exploratory','attributes':ATTRIBUTES,'held_out':held,'seed':2027,'frozen_backbone':False,'trainable_backbone':['blocks.10','blocks.11','norm'],'frozen_head_initialization_sha256':digest(fold/'INIT_HEAD.pt'),'image_augmentation':'horizontal flip probability0.5 only','sampler_alpha':.5,'head_lr':5e-5,'backbone_lr':5e-6,'dropout':.2,'microbatch':16,'accumulate':4,'max_epochs':8,'required_reference_epochs':6,'extension_rule':'epoch6/7 must beat every prior source-selection score by at least1e-4; never extend after epoch8','fixed_learning_rates':True,'weight_decay':.01,'ensemble_siglip_weight':.75,'canvas':[320,160],'backbone':'timm DINOv3 ViT-B/16 FP32 RoPE','foundation_sha256':digest(WEIGHTS/'model.safetensors'),'split_sha256':{p.name:digest(p) for p in split.glob('*.csv')}}
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
    for epoch in range(start,8):
        if not extension_allowed(history,epoch):break
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
    cal_rows=parts['source_calibration'];pcal=predict(ix['source_calibration'])
    cy=np.stack([r[1] for r in cal_rows]);cv=np.stack([r[2] for r in cal_rows]);cd=np.array([r[3] for r in cal_rows])
    old=np.load(SUITE/'siglip_validation_predictions.npz')
    def old_probs(rows):
        lookup_old={str(p):i for i,p in enumerate(old['image_paths'])}
        ids=np.array([lookup_old[r[0]] for r in rows])
        if not np.array_equal(old['y'][ids],np.stack([r[1] for r in rows])) or not np.array_equal(old['valid'][ids],np.stack([r[2] for r in rows])):raise ValueError('SigLIP source validation alignment differs')
        if 'calibration_eligible' in old and not old['calibration_eligible'][ids].all():raise ValueError('Source validation duplicates not excluded')
        return old['prob'][ids]
    mixedcal=.75*old_probs(cal_rows)+.25*pcal
    t=fit_fast(mixedcal,cy,cv,cd);dt=fit_fast(pcal,cy,cv,cd)
    write(fold/'calibration.json',{'attributes':ATTRIBUTES,'thresholds':t.tolist(),'dino_thresholds':dt.tolist(),'source_only':True,'siglip_raw_probability_weight':.75,'calibration_split_sha256':digest(split/'source_calibration.csv')})
    reports={}
    for name,rows,prob in [('selection',selection_rows,selection_prob),('calibration_in_sample',cal_rows,pcal)]:
        yy=np.stack([r[1] for r in rows]);vv=np.stack([r[2] for r in rows]);dd=np.array([r[3] for r in rows]);mix=.75*old_probs(rows)+.25*prob
        np.savez_compressed(fold/(name+'_predictions.npz'),prob=prob,ensemble_prob=mix,y=yy,valid=vv,domains=dd,image_paths=np.array([r[0] for r in rows]))
        reports[name]={'dino_raw':sensitivity(prob,yy,vv,dd,np.full(40,.5)),'dino_calibrated':sensitivity(prob,yy,vv,dd,dt),'ensemble_raw':sensitivity(mix,yy,vv,dd,np.full(40,.5)),'ensemble_calibrated':sensitivity(mix,yy,vv,dd,t)}
    write(fold/'EVALUATION.json',{'candidate':'B final exploratory','best_epoch':ck['epoch'],'epochs_completed':len(history),'source_selection_score':ck['source_selection_score'],'validation_scores':reports,'elapsed_seconds':time.monotonic()-begin,'previous_B_LODO_advancement_gate_passed':False,'separately_user_authorized':True,'hidden_test_used':False,'official_upload_ready':False,'A10_runtime_verified':False})
    write(fold/'LOCK.json',{'files':{p.name:digest(p) for p in [fold/'best.pt',fold/'CONFIG.json',fold/'calibration.json',fold/'selection_predictions.npz',fold/'calibration_in_sample_predictions.npz']},'split_files':{p.name:digest(p) for p in split.glob('*.csv')},'source_only':True})
    write(fold/'PROGRESS.json',{'complete':True,'epochs':len(history),'best_epoch':ck['epoch'],'elapsed_seconds':time.monotonic()-begin})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--probe',action='store_true');p=a.parse_args()
    if p.probe:probe('all_domain')
    else:head('all_domain')
