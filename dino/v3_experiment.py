"""Frozen DINO diagnostic workers. Never trains an all-domain model or uploads."""
import argparse,gc,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from dino.v3_api import DinoV3Backbone,AttributeHead,preprocess,pool_descriptors

BASE=Path('/kaggle/working');SUITE=BASE/'dinov3_results';WEIGHTS=BASE/'dinov3_weights'

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
    def __init__(self,paths,start=0,stop=None):self.paths=paths;self.start=start;self.stop=stop if stop is not None else len(paths)
    def __len__(self):return self.stop-self.start
    def __getitem__(self,i):
        index=self.start+i
        with Image.open(SUITE/'staged_data'/self.paths[index]) as image:p,box=preprocess(image)
        return p,box,index

def extract(start,stop):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.device_count()!=1:raise RuntimeError('Explicit one-GPU extraction affinity required')
    paths=json.loads((SUITE/'INDEX.json').read_text())['paths']
    model=load_backbone().cuda();features=np.load(SUITE/'features.npy',mmap_mode='r+')
    progress=SUITE/f'EXTRACT_{start}.json';begin=time.monotonic()
    loader=DataLoader(Images(paths,start,stop),batch_size=64,num_workers=2,pin_memory=True,timeout=30)
    done=0
    with torch.inference_mode():
        for p,boxes,indices in loader:
            with torch.autocast('cuda',dtype=torch.float16):tokens=model(p.cuda(non_blocking=True))
            descriptors=pool_descriptors(tokens,boxes.cuda()).half().cpu().numpy()
            if not np.isfinite(descriptors).all():raise ValueError('Nonfinite descriptors')
            features[indices.numpy()]=descriptors;done+=len(indices)
            if done%1024<64 or done==stop-start:
                features.flush();write(progress,{'start':start,'stop':stop,'processed':done,'total':stop-start,'elapsed_seconds':time.monotonic()-begin,'completed':done==stop-start})
                print('EXTRACT',start,done,stop-start,flush=True)
    features.flush()

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
    # Fixed features require no target-derived statistics or preprocessing fit.
    cache=np.load(SUITE/'features.npy',mmap_mode='r');features=torch.from_numpy(np.array(cache)).cuda();del cache
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
    config={'attributes':ATTRIBUTES,'held_out':held,'seed':2027,'frozen_backbone':True,'sampler_alpha':.5,'lr':3e-4,'dropout':.2,'batch':128,'max_epochs':30,'patience':5,'min_source_improvement':1e-4,'weight_decay':.01,'ensemble_siglip_weight':.75,'canvas':[320,160],'backbone':'timm DINOv3 ViT-B/16 FP32 RoPE','feature_cache_sha256':json.loads((SUITE/'FEATURE_RECEIPT.json').read_text())['sha256'],'split_sha256':{p.name:digest(p) for p in split.glob('*.csv')}}
    write(fold/'CONFIG.json',config)
    write(fold/'SOURCE_SUPERVISION.json',{'domains':dict(counts),'positive_counts':((sy==1)&sv).sum(0).tolist(),'negative_counts':((sy==0)&sv).sum(0).tolist(),'prevalence':prevalence.tolist(),'positive_weights':wp.cpu().tolist(),'negative_weights':wn.cpu().tolist(),'target_excluded':True})
    model=AttributeHead().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    def predict(indices):
        model.eval();out=[]
        with torch.inference_mode():
            for k in range(0,len(indices),512):
                ii=torch.as_tensor(indices[k:k+512],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):logits=model(features[ii])
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out)
    best=-1.;bad=0;history=[];start=0;seen=np.zeros(len(source),dtype=bool)
    if (fold/'last.pt').exists():
        ck=torch.load(fold/'last.pt',map_location='cpu',weights_only=False)
        if ck['config']!=config:raise ValueError('Resume config mismatch')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'],generator)
        start=ck['epoch']+1;best=ck['best'];bad=ck['bad'];history=ck['history'];seen=ck['seen_indices']
    for epoch in range(start,30):
        tick=time.monotonic();model.train();chosen=torch.multinomial(weights,len(source),replacement=True,generator=generator).numpy()
        seen[chosen]=True
        unique_epoch=Counter(source[j][3] for j in np.unique(chosen));unique_cumulative=Counter(source[j][3] for j in np.flatnonzero(seen))
        draws=Counter(source[j][3] for j in chosen);total_loss=0.;updates=0;skipped=0
        for j in range(0,len(chosen),128):
            batch=torch.as_tensor(ix['source_train'][chosen[j:j+128]],device='cuda');optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=loss_weighted(model(features[batch]),y[batch],v[batch],wp,wn)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite frozen-head loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped+=int(scaler.get_scale()<previous);updates+=1;total_loss+=loss.item()
        prob=predict(ix['source_selection']);rows=parts['source_selection'];score=macro_score(prob,np.stack([r[1] for r in rows]),np.stack([r[2] for r in rows]),[r[3] for r in rows])['macro']['Challenge_Avg']
        row={'epoch':epoch+1,'loss':total_loss/max(1,updates),'source_selection_score':score,'draws':dict(draws),'unique_epoch':dict(unique_epoch),'unique_cumulative':dict(unique_cumulative),'updates':updates,'amp_skips':skipped,'epoch_seconds':time.monotonic()-tick};history.append(row)
        old_best=best
        if score>best:
            best=score;atomic_save({'state_dict':model.state_dict(),'epoch':epoch+1,'source_selection_score':score,'config':config},fold/'best.pt')
        bad=0 if score>old_best+1e-4 else bad+1
        atomic_save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(generator),'epoch':epoch,'best':best,'bad':bad,'history':history,'seen_indices':seen,'config':config},fold/'last.pt')
        write(fold/'HISTORY.json',history);write(fold/'PROGRESS.json',{'epoch':epoch+1,'source_selection_score':score,'elapsed_seconds':time.monotonic()-begin,'complete':False});print(held,json.dumps(row),flush=True)
        if bad>=5:break
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
    a=argparse.ArgumentParser();a.add_argument('--extract',nargs=2,type=int);a.add_argument('--head');p=a.parse_args()
    if p.extract:extract(*p.extract)
    elif p.head:head(p.head)
