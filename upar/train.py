"""Train one SigLIP2-NaFlex semantic attribute model on the UPAR official data.

Run `python -m upar.train --help` for arguments. GPU and internet needed for
first pretrained initialization; resume from local checkpoint afterwards.
"""
from __future__ import annotations
import argparse
import os
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import UPARModel, VisionConfig, set_attribute_queries
from .attributes import ATTRIBUTES, PROMPTS
from .data import UPARDataset, domain_sampler, label_weights
from .metrics import macro_score
from .checkpoint import atomic_save, training_payload, restore_rng
from .parity import code_digest
from .metrics import attribute_report


def loss_weighted(logits, targets, valid, pos_weight, neg_weight):
    valid=valid.bool() & torch.isfinite(targets) & ((targets == 0) | (targets == 1))
    y=torch.where(valid,targets,torch.zeros_like(targets))
    elem=F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    elem=elem*torch.where(y>0.5,pos_weight[None],neg_weight[None])
    return (elem*valid.float()).sum()/valid.float().sum().clamp(min=1)


def build_optimizer(model,lr_backbone,lr_head,weight_decay=0.01):
    groups=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        lr = lr_head if name.startswith('head.') else lr_backbone
        # Layer-wise LR decay across the trainable top four vision blocks.
        if '.encoder.layers.' in name:
            depth=int(name.split('.encoder.layers.',1)[1].split('.',1)[0])
            final_depth=model.cfg.num_hidden_layers-1
            lr=lr_backbone*(0.85**max(0,final_depth-depth))
        decay = 0 if p.ndim==1 or name.endswith('bias') or 'norm' in name.lower() else weight_decay
        groups.append({'params':[p],'lr':lr,'initial_lr':lr,'weight_decay':decay,'is_backbone':not name.startswith('head.')})
    return torch.optim.AdamW(groups)


def load_foundation_model(model,source,device):
    # Hugging Face imported only during initial training, not in evaluator.
    from transformers import Siglip2Model, AutoTokenizer
    foundation=Siglip2Model.from_pretrained(source,attn_implementation='sdpa')
    model.initialize_from_huggingface(foundation)
    tokenizer=AutoTokenizer.from_pretrained(source)
    token=tokenizer(PROMPTS,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
    foundation.text_model.eval().to(device)
    with torch.no_grad():
        features=foundation.get_text_features(**{k:v.to(device) for k,v in token.items()})
    set_attribute_queries(model,features.detach().cpu())
    del foundation
    if device.type=='cuda': torch.cuda.empty_cache()


@torch.inference_mode()
def predict(model,loader,device,amp=True,weights=None):
    model.eval(); outputs=[];labels=[];valids=[];domains=[]
    raw_outputs=[]
    loss_total=0.;valid_total=0
    for patches,mask,shapes,y,v,d in loader:
        patches=patches.to(device,non_blocking=True)
        mask=mask.to(device,non_blocking=True)
        with torch.autocast('cuda',enabled=amp and device.type=='cuda',dtype=torch.float16):
            score=model(patches,mask,shapes)
        if weights is not None:
            count=int(v.sum())
            loss=loss_weighted(score.float(),y.to(device),v.to(device),*weights)
            loss_total+=float(loss)*count;valid_total+=count
        outputs.append(torch.sigmoid(score.float()).cpu().numpy())
        if weights is not None:raw_outputs.append(score.float().cpu().numpy())
        labels.append(y.numpy());valids.append(v.numpy());domains.extend(d)
    result=(np.concatenate(outputs),np.concatenate(labels),np.concatenate(valids),domains)
    return (*result,loss_total/max(1,valid_total),np.concatenate(raw_outputs)) if weights is not None else result


def save_checkpoint(path,model,epoch,metrics,patches,source,**extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    atomic_save({'state_dict':model.state_dict(),'vision_config':model.cfg.as_dict(),
                'max_patches':patches,'model_id':source,'epoch':epoch,'metrics':metrics,
                'attributes':ATTRIBUTES,**extra},path)


def verify_gates(args):
    import hashlib
    data=json.loads(Path(args.data_report).read_text())
    parity=json.loads(Path(args.parity_report).read_text())
    if not data['passed'] or data.get('leakage_review_required',True):
        raise ValueError('Data integrity/duplicate review gate failed')
    for split,path in [('train',args.train_csv),('val',args.val_csv)]:
        if data['splits'][split]['sha256'] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError('Annotation hash changed since audit')
    if not parity['passed'] or parity['code_sha256']!=code_digest() or parity['max_patches']!=args.max_patches:
        raise ValueError('Pretrained parity missing, failed or stale')
    if parity['foundation'] != args.foundation:
        raise ValueError('Foundation differs from parity run')
    if not parity.get('mixed_precision_checked'):
        raise ValueError('GPU mixed-precision parity required before AMP training')
    benchmark=json.loads(Path(args.feasibility_report).read_text())
    if not benchmark.get('passed') or benchmark.get('code_sha256')!=code_digest():
        raise ValueError('GPU feasibility report failed or stale')
    if benchmark.get('parity_report_sha256')!=hashlib.sha256(Path(args.parity_report).read_bytes()).hexdigest():
        raise ValueError('Feasibility report does not match parity evidence')
    if args.scorer_evidence:
        scorer=json.loads(Path(args.scorer_evidence).read_text())
        if not scorer.get('passed') or not scorer.get('organizer_source_sha256'):
            raise ValueError('Provided official scorer verification failed')
    else:
        print('SCORER LIMITATION: published formulas used; organizer implementation unavailable for differential verification.',flush=True)
    return parity


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv',required=True)
    ap.add_argument('--val-csv',required=True)
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--output',default='runs/siglip2')
    ap.add_argument('--foundation',default='google/siglip2-base-patch16-naflex')
    ap.add_argument('--max-patches',type=int,default=256)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--warmup-epochs',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--accumulate',type=int,default=4)
    ap.add_argument('--lr-backbone',type=float,default=1e-5)
    ap.add_argument('--lr-head',type=float,default=1e-4)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--seed',type=int,default=2027)
    ap.add_argument('--ema-decay',type=float,default=0.9995)
    ap.add_argument('--resume',default=None)
    ap.add_argument('--data-report',required=True)
    ap.add_argument('--parity-report',required=True)
    ap.add_argument('--feasibility-report',required=True)
    ap.add_argument('--scorer-evidence')
    ap.add_argument('--max-hours',type=float,default=10.5)
    args=ap.parse_args()
    torch.set_num_threads(int(os.environ.get("UPAR_TORCH_THREADS", "4")))
    parity=verify_gates(args)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required for full training. Use Kaggle/Colab with CUDA.')
    device=torch.device('cuda')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train_ds=UPARDataset(args.train_csv,args.data_root,args.max_patches,augment=True)
    val_ds=UPARDataset(args.val_csv,args.data_root,args.max_patches)
    data_evidence=json.loads(Path(args.data_report).read_text())
    duplicate_val=set(data_evidence.get('duplicate_review',{}).get('val_paths',[]))
    calibration_eligible=np.asarray([row[0] not in duplicate_val for row in val_ds.rows])
    generator=torch.Generator().manual_seed(args.seed)
    train_ld=DataLoader(train_ds,batch_size=args.batch_size,sampler=domain_sampler(train_ds,generator=generator),
                        num_workers=args.workers,pin_memory=True,persistent_workers=False)
    val_ld=DataLoader(val_ds,batch_size=args.batch_size*2,shuffle=False,
                      num_workers=args.workers,pin_memory=True)
    wp,wn,prevalence=label_weights(train_ds)
    wp=wp.to(device);wn=wn.to(device)
    model=UPARModel(VisionConfig(**parity['vision_config']))
    start=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ck.get('schema')!=2:raise ValueError('Resume needs complete schema-2 training state')
        if ck['max_patches'] != args.max_patches: raise ValueError('max-patches must match checkpoint')
        for key in ['epochs','batch_size','accumulate','lr_backbone','lr_head','ema_decay','warmup_epochs','seed','workers']:
            if ck['config'][key]!=getattr(args,key):raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(ck['state_dict'],strict=True)
        start=ck['epoch']+1
    else:
        import hashlib
        prepared_path=Path(args.parity_report).parent/'foundation_native.pt'
        if hashlib.sha256(prepared_path.read_bytes()).hexdigest()!=parity['prepared_sha256']:
            raise ValueError('Prepared foundation changed after numerical parity')
        prepared=torch.load(prepared_path,map_location='cpu',weights_only=False)
        model.load_state_dict(prepared['state_dict'],strict=True)
        del prepared
    model.freeze_vision_except_last(4)
    model=model.to(device)
    optimizer=build_optimizer(model,args.lr_backbone,args.lr_head)
    ema_model=copy.deepcopy(model).eval() if args.ema_decay>0 else None
    if ema_model is not None:
        for p in ema_model.parameters():p.requires_grad_(False)
    def update_ema():
        if ema_model is None:return
        with torch.no_grad():
            for ep,sp in zip(ema_model.parameters(),model.parameters()):
                ep.lerp_(sp,1-args.ema_decay)
            for eb,sb in zip(ema_model.buffers(),model.buffers()):
                eb.copy_(sb)
    scaler=torch.amp.GradScaler('cuda')
    steps_per_epoch=math.ceil(len(train_ld)/args.accumulate)
    total_steps=max(1,steps_per_epoch*args.epochs)
    warm_steps=max(1,int(total_steps*.05))
    step=steps_per_epoch*start
    best=-1
    if args.resume:
        optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        if ema_model is not None:ema_model.load_state_dict(ck['ema_state_dict'],strict=True)
        best=ck['best_score'];step=ck['step'];restore_rng(ck['rng'],generator)
        # A hard stop can occur after best.pt was atomically written but before
        # last.pt. Never overwrite that newer best using an older resume score.
        historical_best=out/'best.pt'
        if historical_best.exists():
            previous=torch.load(historical_best,map_location='cpu',weights_only=False)
            if previous.get('attributes')!=ATTRIBUTES:raise ValueError('Existing best has different attributes')
            best=max(best,float(previous['metrics']['macro']['Challenge_Avg']))
            del previous
    run_started=time.monotonic()
    with open(out/'training_config.json','w') as f: json.dump(vars(args),f,indent=2)
    with open(out/'prevalence.json','w') as f:json.dump(dict(zip(ATTRIBUTES,prevalence.tolist())),f,indent=2)
    for epoch in range(start,args.epochs):
        t0=time.time()
        # Keep optimizer groups and use LR=0 for vision during head warmup.
        model.train(); optimizer.zero_grad(set_to_none=True)
        model.freeze_vision_except_last(4 if epoch >= args.warmup_epochs else 0)
        train_loss=0.; n=0; images_epoch=0; successful_updates=0; skipped_updates=0
        for k,(patches,mask,shapes,y,v,_) in enumerate(train_ld):
            patches=patches.to(device,non_blocking=True)
            mask=mask.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True);v=v.to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(patches,mask,shapes)
                loss=loss_weighted(logits,y,v,wp,wn)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, batch {k}; last completed checkpoint retained')
            group_size=min(args.accumulate,len(train_ld)-(k//args.accumulate)*args.accumulate)
            scaler.scale(loss/group_size).backward()
            train_loss+=loss.item();n+=1
            images_epoch+=len(y)
            if (k+1)%args.accumulate==0 or k+1==len(train_ld):
                step+=1
                factor=(step/warm_steps) if step<warm_steps else (.5*(1+math.cos(math.pi*(step-warm_steps)/max(1,total_steps-warm_steps))))
                for g in optimizer.param_groups:
                    g['lr']=g['initial_lr']*max(factor,.01)*(0 if epoch<args.warmup_epochs and g['is_backbone'] else 1)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                previous_scale=scaler.get_scale()
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale()>=previous_scale:
                    update_ema();successful_updates+=1
                else:skipped_updates+=1
                if step%50==0 or k+1==len(train_ld):
                    elapsed=time.monotonic()-run_started
                    progress={'epoch':epoch+1,'epochs':args.epochs,'optimizer_updates_attempted':step,
                        'optimizer_updates_total':total_steps,'images_processed':epoch*len(train_ds)+images_epoch,
                        'elapsed_seconds':elapsed,'training_started':True}
                    temporary=out/'progress.tmp.json';temporary.write_text(json.dumps(progress));temporary.replace(out/'progress.json')
                    print('PROGRESS '+json.dumps(progress),flush=True)
        train_seconds=time.time()-t0
        alternatives=[]
        for kind,candidate in [('raw',model),('ema',ema_model)]:
            if candidate is None:continue
            prob,true,valid,domains,val_loss,logits=predict(candidate,val_ld,device,weights=(wp,wn))
            candidate_report=macro_score(prob,true,valid,domains)
            candidate_report['validation_loss']=val_loss
            alternatives.append((candidate_report['macro']['Challenge_Avg'],kind,candidate,prob,true,valid,domains,candidate_report,logits))
        cur,selected,scoring_model,val_prob,val_true,val_valid,val_domains,report,val_logits=max(alternatives,key=lambda x:x[0])
        report['duplicate_excluded_sensitivity']=macro_score(val_prob[calibration_eligible],val_true[calibration_eligible],
                    val_valid[calibration_eligible],np.asarray(val_domains)[calibration_eligible].tolist())
        cur=report['macro']['Challenge_Avg']
        print(f'Epoch {epoch+1}/{args.epochs} loss={train_loss/max(1,n):.5f} val_macro_avg={cur:.5f} time={time.time()-t0:.1f}s',flush=True)
        with open(out/'history.jsonl','a') as f:
            f.write(json.dumps({'epoch':epoch+1,'train_loss':train_loss/max(1,n),'metrics':report,
                                'validation_candidates':{z[1]:z[7] for z in alternatives},'selected':selected,
                                'images_processed_epoch':images_epoch,'images_processed_cumulative':(epoch+1)*len(train_ds),
                                'optimizer_updates_attempted_cumulative':step,'optimizer_updates_successful_epoch':successful_updates,
                                'amp_skipped_updates_epoch':skipped_updates,'training_seconds':train_seconds,
                                'epoch_seconds':time.time()-t0})+'\n')
        if cur>best:
            best=cur
            save_checkpoint(out/'best.pt',scoring_model,epoch,report,args.max_patches,args.foundation,
                            trained=True,selected=selected,parity_code_sha256=parity['code_sha256'])
            np.savez_compressed(out/'best_val_predictions.npz',prob=val_prob,logits=val_logits,y=val_true,
                                valid=val_valid,domains=np.asarray(val_domains),
                                calibration_eligible=calibration_eligible,
                                image_paths=np.asarray([row[0] for row in val_ds.rows]))
            with open(out/'best_val_metrics.json','w') as f:json.dump(report,f,indent=2)
            with open(out/'attribute_metrics.json','w') as f:
                json.dump({'attributes':ATTRIBUTES,**attribute_report(val_prob,val_true,val_valid)},f,indent=2)
            qualitative=[]
            for domain in sorted(set(val_domains)):
                indices=np.flatnonzero(np.asarray(val_domains)==domain)
                for index in indices[np.linspace(0,len(indices)-1,3,dtype=int)]:
                    qualitative.append({'image':val_ds.rows[index][0],'domain':domain,
                        'probabilities':val_prob[index].tolist(),'labels':val_true[index].tolist(),
                        'valid':val_valid[index].tolist()})
            (out/'qualitative_predictions.json').write_text(json.dumps({'attributes':ATTRIBUTES,'examples':qualitative},indent=2))
        # Keep raw training state separate from inference/best weights.
        atomic_save(training_payload(model,ema_model,optimizer,scaler,generator,epoch,step,best,vars(args),
                    max_patches=args.max_patches,attributes=ATTRIBUTES),out/'last.pt')
        if time.monotonic()-run_started > args.max_hours*3600:
            print('Stopping at epoch boundary for persistent checkpoint/resume',flush=True)
            break
    print(f'Finished. Best score observed in run: {best:.5f}; calibrate held-out probabilities next.')

if __name__=='__main__':main()
