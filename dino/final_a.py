"""Exploratory candidate A: one all-domain frozen head, never an upload."""
import hashlib,json,os,random,shutil,subprocess,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from dino.model import AttributeHead
from dino.v3_experiment import digest,write
from upar.attributes import ATTRIBUTES
from upar.train import loss_weighted
from upar.metrics import macro_score
from upar.checkpoint import atomic_save,rng_state,restore_rng
from scripts.generalization_analysis import fit_fast,sensitivity

ROOT=Path('/kaggle/working');OUT=ROOT/'candidate_a';FROZEN_CACHE_SHA='bd7991deafba88f07edd0d53525252b20989a52ac2e355846b4ce8aa62a2285d'
INDEX_SHA='ce8da15c246c7c7ff4cc7db5055d08d29351f9f5c832e9ea8adeb4bfe7756ac6'
LABEL_SHA='b54f333d62ae3c4b521108dbe98fd2e347f196bf3d2e6531e412d456c7ed399d'

def find(name):
    results=[]
    for root,dirs,files in os.walk('/kaggle/input'):
        dirs[:]=[d for d in dirs if d not in {'staged_data','archive_cache','official_data','hf-cache','upar_code','submission_candidate'}]
        if name in files:results.append(Path(root)/name)
    return results

def split_validation(index,ntrain=97669):
    """Predeclared seed-hash 50/50 split of pixel groups, no label fitting."""
    hashes=index['audited_pixel_sha256'];training=set(hashes[:ntrain])
    eligible=np.array([i for i in range(ntrain,len(hashes)) if hashes[i] not in training],dtype=np.int64)
    selected=np.array([int(hashlib.sha256(('candidate-a-2027:'+hashes[i]).encode()).hexdigest()[:16],16)%2==0 for i in eligible])
    selection,calibration=eligible[selected],eligible[~selected]
    if not len(selection) or not len(calibration):raise ValueError('Empty source partition')
    if set(hashes[i] for i in selection)&set(hashes[i] for i in calibration):raise ValueError('Pixel overlap')
    return selection,calibration

def copy_cache(source):
    OUT.mkdir(exist_ok=True)
    for name,sha in [('features.npy',FROZEN_CACHE_SHA),('INDEX.json',INDEX_SHA),('labels.npz',LABEL_SHA)]:
        dest=OUT/name;shutil.copyfile(Path(source)/name,dest)
        if digest(dest)!=sha:raise ValueError('Frozen cache bytes changed: '+name)

def main():
    OUT.mkdir(exist_ok=True);begin=time.monotonic()
    def status(phase,**kw):write(ROOT/'STATUS.json',{'candidate':'A','phase':phase,'elapsed_seconds':time.monotonic()-begin,'competition_uploads':0,**kw})
    try:
        status('cache_integrity',training_started=False)
        torch.set_num_threads(2);torch.manual_seed(2027);np.random.seed(2027);random.seed(2027)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('Need 4GiB free')
        caches=find('FEATURE_RECEIPT.json');caches=[p for p in caches if p.parent.name=='dinov3_results']
        if len(caches)!=1:raise ValueError('Require unique verified frozen cache')
        source=caches[0].parent;receipt=json.loads(caches[0].read_text())
        if receipt['sha256']!=FROZEN_CACHE_SHA or not receipt['complete']:raise ValueError('Cache receipt mismatch')
        # Copy three compact files once; never enumerate or copy mounted images.
        subprocess.run([sys.executable,'-m','dino.final_a','--cache',str(source)],check=True,timeout=120)
        if time.monotonic()-begin>120:raise TimeoutError('Cache setup exceeded120s')
        index=json.loads((OUT/'INDEX.json').read_text());labels=np.load(OUT/'labels.npz')
        if len(index['paths'])!=131076 or len(set(index['paths']))!=131076:raise ValueError('Unexpected cache identity')
        ntrain=97669;train=np.arange(ntrain);selection,cal=split_validation(index,ntrain)
        counts=Counter(labels['domains'][:ntrain]);expected={'PA100k':79001,'PETA':8668,'Market1501':10000}
        if counts!=expected:raise ValueError('Official training domain counts changed')
        config={'candidate':'A','status':'exploratory_not_previously_promoted','seed':2027,'foundation_sha256':'1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332','feature_sha256':FROZEN_CACHE_SHA,'attributes':ATTRIBUTES,'frozen_backbone':True,'sampler_alpha':.5,'head_lr':3e-4,'weight_decay':.01,'dropout':.2,'batch':128,'max_epochs':30,'patience':5,'min_improvement':1e-4,'siglip_raw_probability_weight':.75,'selection_rows':len(selection),'calibration_rows':len(cal),'excluded_train_pixel_duplicate_validation_rows':131076-ntrain-len(selection)-len(cal),'validation_split':'SHA256(candidate-a-2027:+audited_pixel_sha256) first64bits mod2; 0 selection,1 calibration'}
        write(OUT/'CONFIG.json',config)
        np.savez_compressed(OUT/'PARTITIONS.npz',source_train=train,source_selection=selection,source_calibration=cal)
        for name,ii in [('source_selection',selection),('source_calibration',cal)]:
            write(OUT/(name+'_MANIFEST.json'),{'paths':[index['paths'][i] for i in ii],'pixel_sha256':[index['audited_pixel_sha256'][i] for i in ii]})
        data=np.load(OUT/'features.npy',mmap_mode='r')
        if data.shape!=(131076,6,768) or data.dtype!=np.float16:raise ValueError('Unexpected feature shape')
        features=torch.from_numpy(np.array(data)).cuda();del data
        y=torch.from_numpy(labels['y']).cuda();v=torch.from_numpy(labels['valid']).cuda()
        sy=labels['y'][:ntrain];sv=labels['valid'][:ntrain].astype(bool)
        prev=np.divide(((sy==1)&sv).sum(0),sv.sum(0),out=np.full(40,.5),where=sv.sum(0)>0)
        wp=torch.tensor(np.exp(1-prev).astype(np.float32),device='cuda');wn=torch.tensor(np.exp(prev).astype(np.float32),device='cuda')
        write(OUT/'SOURCE_SUPERVISION.json',{'domains':dict(counts),'positives':((sy==1)&sv).sum(0).tolist(),'negatives':((sy==0)&sv).sum(0).tolist(),'prevalence':prev.tolist(),'class_frequency_unweighted':True})
        weights=torch.tensor([counts[d]**-.5 for d in labels['domains'][:ntrain]],dtype=torch.double)
        generator=torch.Generator().manual_seed(2027);model=AttributeHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
        def predict(ii):
            model.eval();out=[]
            with torch.inference_mode():
                for start in range(0,len(ii),512):
                    ix=torch.as_tensor(ii[start:start+512],device='cuda')
                    with torch.autocast('cuda',dtype=torch.float16):z=model(features[ix])
                    out.append(z.float().sigmoid().cpu().numpy())
            return np.concatenate(out)
        history=[];best=-1.;bad=0;seen=np.zeros(ntrain,bool);start=0
        if (OUT/'last.pt').exists():
            ck=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
            if ck['config']!=config:raise ValueError('Resume config changed')
            model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'],generator)
            start=ck['epoch']+1;best=ck['best'];bad=ck['bad'];history=ck['history'];seen=ck['seen']
        status('frozen_head_training',training_started=True)
        write(ROOT/'TRAINING_START.json',{'candidate':'A','cache_verified':True,'backbone_trainable':False})
        for epoch in range(start,30):
            tick=time.monotonic();chosen=torch.multinomial(weights,ntrain,replacement=True,generator=generator).numpy();seen[chosen]=True;model.train();loss_sum=0.;skips=0;updates=0
            for j in range(0,ntrain,128):
                ix=torch.as_tensor(chosen[j:j+128],device='cuda');opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=loss_weighted(model(features[ix]),y[ix],v[ix],wp,wn)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                oldscale=scaler.get_scale();scaler.step(opt);scaler.update();skips+=int(scaler.get_scale()<oldscale);updates+=1;loss_sum+=loss.item()
            p=predict(selection);score=macro_score(p,labels['y'][selection],labels['valid'][selection],labels['domains'][selection])['macro']['Challenge_Avg']
            row={'epoch':epoch+1,'loss':loss_sum/updates,'source_selection_score':score,'seconds':time.monotonic()-tick,'updates':updates,'amp_skips':skips,'draws':dict(Counter(labels['domains'][chosen])),'unique_epoch':dict(Counter(labels['domains'][np.unique(chosen)])),'unique_cumulative':dict(Counter(labels['domains'][:ntrain][seen]))};history.append(row)
            previous=best
            if score>best:
                best=score;atomic_save({'state_dict':model.state_dict(),'epoch':epoch+1,'config':config,'source_selection_score':score},OUT/'best.pt')
            bad=0 if score>previous+1e-4 else bad+1
            atomic_save({'state_dict':model.state_dict(),'epoch':epoch,'config':config,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(generator),'history':history,'best':best,'bad':bad,'seen':seen},OUT/'last.pt')
            write(OUT/'HISTORY.json',history);print(json.dumps(row),flush=True)
            if time.monotonic()-begin>1200:raise TimeoutError('Candidate A head job cap1200s')
            if bad>=5:break
        ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'])
        val=np.arange(ntrain,131076);pval=predict(val)
        candidates=find('best_val_predictions.npz');matched=[]
        for path in candidates:
            with np.load(path) as z:
                if 'image_paths' in z and len(z['prob'])==33407:matched.append(path)
        if len(matched)!=1:raise ValueError('Unique original all-domain SigLIP validation predictions required')
        old=np.load(matched[0]);paths=np.asarray(index['paths'][ntrain:])
        if not np.array_equal(old['image_paths'],paths) or not np.array_equal(old['y'],labels['y'][ntrain:]) or not np.array_equal(old['valid'],labels['valid'][ntrain:]):raise ValueError('Original SigLIP validation identity differs')
        mix=.75*old['prob']+.25*pval;ci=cal-ntrain;si=selection-ntrain
        if 'calibration_eligible' in old and not old['calibration_eligible'][ci].all():raise ValueError('Original duplicate eligibility excludes calibration rows')
        t=fit_fast(mix[ci],labels['y'][cal],labels['valid'][cal],labels['domains'][cal]);dt=fit_fast(pval[ci],labels['y'][cal],labels['valid'][cal],labels['domains'][cal])
        write(OUT/'calibration.json',{'attributes':ATTRIBUTES,'thresholds':t.tolist(),'dino_thresholds':dt.tolist(),'source_only':True,'siglip_raw_probability_weight':.75,'calibration_partition_sha256':digest(OUT/'source_calibration_MANIFEST.json')})
        np.savez_compressed(OUT/'validation_predictions.npz',prob=pval,ensemble_prob=mix,y=labels['y'][ntrain:],valid=labels['valid'][ntrain:],domains=labels['domains'][ntrain:],image_paths=paths,selection_indices=si,calibration_indices=ci)
        report={'best_epoch':ck['epoch'],'source_selection_score':ck['source_selection_score'],'elapsed_seconds':time.monotonic()-begin,'siglip_validation_predictions_sha256':digest(matched[0]),'runtime_A10_verified':False,'official_upload_ready':False,'previous_LODO_primary_promotion_passed':False,'validation_scores':{}}
        for name,p,threshold in [('siglip_raw',old['prob'],np.full(40,.5)),('dino_raw',pval,np.full(40,.5)),('ensemble_raw',mix,np.full(40,.5)),('ensemble_calibrated',mix,t)]:
            report['validation_scores'][name]={'selection':sensitivity(p[si],labels['y'][selection],labels['valid'][selection],labels['domains'][selection],threshold),'calibration_in_sample':sensitivity(p[ci],labels['y'][cal],labels['valid'][cal],labels['domains'][cal],threshold)}
        write(OUT/'EVALUATION.json',report)
        write(OUT/'LOCK.json',{'files':{p.name:digest(p) for p in [OUT/'best.pt',OUT/'CONFIG.json',OUT/'PARTITIONS.npz',OUT/'calibration.json',OUT/'validation_predictions.npz']},'exploratory_candidate_only':True})
        write(OUT/'CACHE_REUSE_RECEIPT.json',{'original_cache_sha256':FROZEN_CACHE_SHA,'index_sha256':INDEX_SHA,'labels_sha256':LABEL_SHA,'images_extracted':0,'cache_reused':True})
        status('complete',head_complete=True,offline_package_benchmark_pending=True)
    except BaseException as e:
        status('failed',error=type(e).__name__+': '+str(e));raise

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--cache':copy_cache(sys.argv[2])
    else:main()
