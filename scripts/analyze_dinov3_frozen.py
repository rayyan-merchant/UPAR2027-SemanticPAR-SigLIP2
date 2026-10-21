"""Independent CPU audit of DINOv3 frozen LODO; never reads hidden-test data."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from upar.attributes import ATTRIBUTES
from upar.data import load_rows
from scripts.generalization_analysis import fit_fast

FOLDS=['PA100k','PETA','Market1501']
GROUPS={'age':list(range(3)),'gender':[3],'hair':[4,5,6],'upper_length':[7],'upper_colors':list(range(8,20)),'lower_length':[20],'lower_colors':list(range(21,33)),'lower_type':[33,34],'accessories':list(range(35,40))}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def safe(n,d):return np.divide(n,d,out=np.zeros_like(np.asarray(n),dtype=float),where=np.asarray(d)!=0)
def harmonic(a,b):return 2*a*b/(a+b) if a+b else 0.
def domain(p,y,v,t):
    mask=v.astype(bool);true=y==1;pred=p>np.asarray(t)
    tp=(pred&true&mask).sum(0);tn=(~pred&~true&mask).sum(0);fp=(pred&~true&mask).sum(0);fn=(~pred&true&mask).sum(0)
    pos=tp+fn;neg=tn+fp;both=(pos>0)&(neg>0);known=(pos+neg)>0
    am=(safe(tp,pos)+safe(tn,neg))/2;lf=safe(2*tp,2*tp+fp+fn)
    inter=(pred&true&mask).sum(1);pc=(pred&mask).sum(1);gc=(true&mask).sum(1)
    precision=float(safe(inter,pc).mean());recall=float(safe(inter,gc).mean());f1=harmonic(precision,recall);ma=float(am[both].mean()) if both.any() else 0.
    mif=float(safe(2*inter,pc+gc).mean());allma=float(am[known].mean()) if known.any() else 0.
    metrics={'Challenge_Avg':harmonic(ma,f1),'mA':ma,'Inst_F1':f1,'Inst_Prec':precision,'Inst_Rec':recall,'Inst_Acc':float(safe(inter,pc+gc-inter).mean()),'Label_F1':float(lf[known].mean()) if known.any() else 0.,'mean_image_F1':mif,'mean_image_Challenge_Avg':harmonic(ma,mif),'all_valid_safe_zero_mA':allma,'all_valid_safe_zero_mean_image_Challenge_Avg':harmonic(allma,mif)}
    attrs=[{'attribute':ATTRIBUTES[j],'tp':int(tp[j]),'tn':int(tn[j]),'fp':int(fp[j]),'fn':int(fn[j]),'positives':int(pos[j]),'negatives':int(neg[j]),'mA':float(am[j]) if both[j] else None,'label_F1':float(lf[j]),'precision':float(safe(tp[j],tp[j]+fp[j])),'recall':float(safe(tp[j],pos[j])),'threshold':float(t[j])} for j in range(40)]
    return metrics,attrs
def macro(p,y,v,d,t):
    values=[domain(p[d==name],y[d==name],v[d==name],t)[0] for name in sorted(set(d))]
    return {k:float(np.mean([x[k] for x in values])) for k in values[0]}
def group(attrs):
    return {n:{'mA':float(np.mean([attrs[i]['mA'] for i in indices if attrs[i]['mA'] is not None])) if any(attrs[i]['mA'] is not None for i in indices) else None,'label_F1':float(np.mean([attrs[i]['label_F1'] for i in indices]))} for n,indices in GROUPS.items()}
def dump(path,v):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifacts',type=Path,required=True);ap.add_argument('--heads',type=Path,required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[1];suite=a.artifacts;out=root/'reports/dino_v3/results';out.mkdir(exist_ok=True)
    baseline=root.parent/'lodo/threefold-v2-results/lodo';plan=read(root/'reports/dino_v3/PLAN.json');half=np.full(40,.5,np.float32);summary=read(suite/'CONSOLIDATED.json')
    lock=read(root/'reports/lodo/equal_domain/BASELINE_LOCK.json')['files']
    for p,h in lock.items():assert sha(root/p)==h,('baseline changed',p)
    init=read(suite/'INITIALIZATION_RECEIPT.json');cache=read(suite/'FEATURE_RECEIPT.json');assert init['pretrained_only'] and not init['PAR_weights_loaded'];assert cache['complete'] and cache['shape']==[131076,6,768] and cache['no_global_target_statistics_fit']
    assert cache['backbone_sha256']==init['foundation_sha256']==plan['foundation_sha256'];assert sha(root/'dino/v3_model.py')==cache['native_code_sha256']
    for n in ['GATE_FEASIBILITY.json','GATE_PARITY.json','GATE_HARDWARE.json']:assert sha(root/'reports/dino_v3'/n)==plan['gpu_gate_sha256'][n]
    checks={};allmetrics=[];allattrs=[];allgroups=[];curves={}
    for held in FOLDS:
        f=suite/held;runtime=read(f/'EVALUATION.json');cfg=read(f/'CONFIG.json');cal=read(f/'CALIBRATION.json');locked=read(f/'LOCK.json')
        assert locked['source_only_selection_and_calibration'] and locked['target_evaluation_count']==runtime['target_evaluations']==1
        for p,h in locked['files'].items():
            path=(a.heads/p) if p.endswith('.pt') else (suite/p);assert sha(path)==h,('lock mismatch',p)
        assert sha(f/'target_predictions.npz')==runtime['target_prediction_sha256'] and sha(f/'LOCK.json')==runtime['lock_sha256']
        ck=torch.load(a.heads/held/'best.pt',map_location='cpu',weights_only=False);assert ck['config']==cfg and ck['epoch']==runtime['best_epoch'] and ck['source_selection_score']==runtime['source_selection_score']
        assert cfg['attributes']==ATTRIBUTES and cfg['sampler_alpha']==.5 and cfg['frozen_backbone'] and cfg['seed']==2027 and cfg['canvas']==[320,160] and cfg['feature_cache_sha256']==cache['sha256']
        parts={}
        for name,info in plan['expected_splits'][held]['partitions'].items():
            p=f/'splits'/(name+'.csv');assert sha(p)==info['sha256']==cfg['split_sha256'][p.name];parts[name]=load_rows(p)
        names=['source_train','source_selection','source_calibration','sealed_target_evaluation'];sets=[{r[0] for r in parts[n]} for n in names]
        for i in range(4):
            for j in range(i):assert not sets[i]&sets[j]
        for n in names[:3]:assert held not in {r[3] for r in parts[n]}
        predictions={}
        for split,file in [('source_selection','source_selection_predictions.npz'),('source_calibration','source_calibration_predictions.npz'),('sealed_target_evaluation','target_predictions.npz')]:
            z=np.load(f/file);rows=parts[split];assert np.array_equal(z['image_paths'],np.array([r[0] for r in rows]));assert np.array_equal(z['y'],np.stack([r[1] for r in rows]));assert np.array_equal(z['valid'],np.stack([r[2] for r in rows]));assert np.array_equal(z['domains'],np.array([r[3] for r in rows]));predictions[split]=dict(z)
        oldcal=np.load(baseline/held/'source_calibration_predictions.npz');oldtarget=np.load(baseline/held/'target_predictions.npz');pc=predictions['source_calibration'];z=predictions['sealed_target_evaluation'];selection=predictions['source_selection']
        for name,old in [('source_calibration_predictions.npz',oldcal),('target_predictions.npz',oldtarget)]:
            assert sha(baseline/held/name)==plan['baseline_predictions'][held][name]
            new=pc if name.startswith('source') else z
            for key in ['y','valid','domains']:assert np.array_equal(old[key],new[key])
        tc=np.asarray(cal['dino_thresholds'],np.float32);te=np.asarray(cal['ensemble_thresholds'],np.float32);assert not cal['target_used'] and cal['ensemble_weight_siglip']==.75 and held not in cal['source_domains']
        mixedcal=.75*oldcal['prob']+.25*pc['prob'];mixed=.75*oldtarget['prob']+.25*z['prob'];np.testing.assert_array_equal(mixed,z['ensemble_prob'])
        np.testing.assert_array_equal(fit_fast(pc['prob'],pc['y'],pc['valid'],pc['domains']),tc);np.testing.assert_array_equal(fit_fast(mixedcal,pc['y'],pc['valid'],pc['domains']),te)
        baseline_eval=read(baseline/held/'EVALUATION.json');old_threshold=np.asarray(read(baseline/held/'calibration.json')['thresholds'],np.float32) if (baseline/held/'calibration.json').exists() else None
        if old_threshold is None:
            # Baseline lock records its unchanged calibrated40threshold vector.
            candidates=list((baseline/held).rglob('calibration.json'));assert len(candidates)==1;old_threshold=np.asarray(read(candidates[0])['thresholds'],np.float32)
        conditions={'siglip_raw':(oldtarget['prob'],half),'siglip_calibrated':(oldtarget['prob'],old_threshold),'dino_raw':(z['prob'],half),'dino_calibrated':(z['prob'],tc),'ensemble_raw':(mixed,half),'ensemble_calibrated':(mixed,te)}
        foldmetrics={}
        for setting,(prob,t) in conditions.items():
            metrics,attrs=domain(prob,z['y'],z['valid'],t);foldmetrics[setting]=metrics
            if setting in runtime['reports']:
                recorded=runtime['reports'][setting]['legacy']['macro']
                for k,v in recorded.items():assert abs(metrics[k]-v)<1e-10,(held,setting,k,metrics[k],v)
                assert abs(metrics['mean_image_Challenge_Avg']-runtime['reports'][setting]['mean_image_f1_sensitivity']['macro']['Challenge_Avg'])<1e-10
            for row in attrs:allattrs.append(dict(domain=held,setting=setting,**row))
            for n,value in group(attrs).items():allgroups.append(dict(domain=held,setting=setting,group=n,**value))
            allmetrics.append(dict(domain=held,setting=setting,**metrics))
        assert abs(foldmetrics['siglip_calibrated']['Challenge_Avg']-plan['baseline_calibrated'][held])<1e-10
        source_raw=macro(selection['prob'],selection['y'],selection['valid'],selection['domains'],half);assert abs(source_raw['Challenge_Avg']-runtime['source_selection_score'])<1e-10
        source_cal=macro(selection['prob'],selection['y'],selection['valid'],selection['domains'],tc)
        # Baseline selection arrays are independently checked against the same sealed selection CSV.
        oldsel=np.load(baseline/held/'training/best_val_predictions.npz')
        for key in ['y','valid','domains']:assert np.array_equal(oldsel[key],selection[key]),('source selection alignment',held,key)
        mixedselection=.75*oldsel['prob']+.25*selection['prob'];ensemble_source=macro(mixedselection,selection['y'],selection['valid'],selection['domains'],te)
        history=read(f/'HISTORY.json');assert len(history)==runtime['epochs'] and 1<=len(history)<=30;assert max(x['source_selection_score'] for x in history)==runtime['source_selection_score']
        source=parts['source_train'];counts={d:sum(r[3]==d for r in source) for d in sorted({r[3] for r in source})};draws={d:0 for d in counts}
        for row in history:
            assert row['updates']==math.ceil(len(source)/128) and sum(row['draws'].values())==len(source)
            assert 0<=row['amp_skips']<=row['updates']
            for d,n in row['draws'].items():
                assert d in counts and 0<=row['unique_epoch'][d]<=min(n,counts[d]) and row['unique_epoch'][d]<=row['unique_cumulative'][d]<=counts[d];draws[d]+=n
        expected={d:math.sqrt(n)/sum(math.sqrt(q) for q in counts.values()) for d,n in counts.items()};exposure={d:draws[d]/sum(draws.values()) for d in draws}
        for d in draws:assert abs(exposure[d]-expected[d])<.01
        sy=np.stack([r[1] for r in source]);sv=np.stack([r[2] for r in source]);positive=((sy==1)&sv).sum(0);negative=((sy==0)&sv).sum(0);supervision=read(f/'SOURCE_SUPERVISION.json');assert positive.tolist()==supervision['positive_counts'] and negative.tolist()==supervision['negative_counts']
        prevalence=np.divide(positive,sv.sum(0),out=np.full(40,.5),where=sv.sum(0)>0);np.testing.assert_allclose(np.exp(1-prevalence),supervision['positive_weights'],rtol=1e-6);np.testing.assert_allclose(np.exp(prevalence),supervision['negative_weights'],rtol=1e-6)
        mask=z['valid'].astype(bool);corr=float(np.corrcoef((oldtarget['prob']-z['y'])[mask],(z['prob']-z['y'])[mask])[0,1]);assert abs(corr-runtime['error_complementarity']['raw_probability_residual_correlation'])<1e-12
        curves[held]={'history':history,'attempted_updates':sum(x['updates'] for x in history),'successful_updates':sum(x['updates']-x['amp_skips'] for x in history),'amp_skips':sum(x['amp_skips'] for x in history),'images_consumed':len(source)*len(history),'best_epoch':runtime['best_epoch'],'epochs':len(history),'elapsed_seconds':runtime['elapsed_seconds'],'source_selection_raw':source_raw,'source_selection_calibrated':source_cal,'ensemble_source_selection_calibrated':ensemble_source,'dino_source_target_calibrated_gap':source_cal['Challenge_Avg']-foldmetrics['dino_calibrated']['Challenge_Avg'],'ensemble_source_target_calibrated_gap':ensemble_source['Challenge_Avg']-foldmetrics['ensemble_calibrated']['Challenge_Avg'],'source_counts':counts,'source_positive_counts':positive.tolist(),'source_negative_counts':negative.tolist(),'source_weak_positives_20_or_fewer':{ATTRIBUTES[i]:int(n) for i,n in enumerate(positive) if n<=20},'source_missing_positive_attributes':[ATTRIBUTES[i] for i,n in enumerate(positive) if n==0],'exposure_expected':expected,'exposure_actual':exposure,'unique_seen_final':history[-1]['unique_cumulative'],'error_complementarity':runtime['error_complementarity'],'calibration':{'dino_shift_below_half':int((tc<.5).sum()),'dino_shift_above_half':int((tc>.5).sum()),'ensemble_below_half':int((te<.5).sum()),'ensemble_above_half':int((te>.5).sum())}}
        checks[held]={'checkpoint_config_calibration_split_locks_verified':True,'source_only_threshold_refits_exact':True,'independent_metrics_match':True,'row_identity_and_source_isolation_verified':True,'source_frequency_BCE_verified':True,'exposure_consistent':True,'target_evaluations_recorded':1,'checkpoint_sha256':sha(a.heads/held/'best.pt'),'locked_file_count':len(locked['files'])}
    totals={}
    for setting in ['siglip_raw','siglip_calibrated','dino_raw','dino_calibrated','ensemble_raw','ensemble_calibrated']:
        rows=[x for x in allmetrics if x['setting']==setting];totals[setting]={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ['domain','setting']};totals[setting]['worst_Challenge_Avg']=min(x['Challenge_Avg'] for x in rows);totals[setting]['worst_mean_image_Challenge_Avg']=min(x['mean_image_Challenge_Avg'] for x in rows)
        if setting in summary['summary']:
            assert abs(totals[setting]['Challenge_Avg']-summary['summary'][setting]['macro'])<1e-10 and abs(totals[setting]['worst_Challenge_Avg']-summary['summary'][setting]['worst'])<1e-10
    promotion={'macro_at_least_.8023':totals['ensemble_calibrated']['Challenge_Avg']>=.8023,'worst_at_least_.7772':totals['ensemble_calibrated']['worst_Challenge_Avg']>=.7772,'no_domain_regression_over_.005':all(next(x for x in allmetrics if x['domain']==d and x['setting']=='ensemble_calibrated')['Challenge_Avg']>=plan['baseline_calibrated'][d]-.005 for d in FOLDS)}
    assert all(promotion.values())==summary['ensemble_promotion_criteria_met']
    dump(out/'INDEPENDENT_VERIFICATION.json',{'passed':True,'folds':checks,'protected_baseline_files_verified':len(lock),'cache_payload_not_downloaded_or_independently_rehashed':True,'cache_receipt_identity_verified':True,'metrics_conventions_uncertain_official_edge_cases':True,'promotion':promotion,'promoted':all(promotion.values())})
    dump(out/'SUMMARY.json',{'settings':totals,'fold_metrics':allmetrics,'promotion':promotion,'promoted':all(promotion.values()),'session_runner_seconds':summary['elapsed_seconds']})
    dump(out/'SOURCE_CURVES_SUPPORT.json',curves)
    for filename,rows in [('METRICS.csv',allmetrics),('ATTRIBUTE_METRICS.csv',allattrs),('GROUP_METRICS.csv',allgroups)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'independent_checks_passed':True,'summary':totals,'promotion':promotion},indent=2))

if __name__=='__main__':main()
