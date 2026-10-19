"""CPU-only diagnosis using known official validation labels; never fits to test results."""
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from upar.metrics import macro_score,calibrate_thresholds
from upar.attributes import ATTRIBUTES

GRID=np.asarray([.22,.28,.34,.40,.45,.50,.55,.60,.66,.72,.78],np.float32)
def div(a,b):return np.divide(a,b,out=np.zeros_like(np.asarray(a),dtype=np.float64),where=np.asarray(b)!=0)
def harmonic(a,b):return 2*a*b/(a+b) if a+b else 0.

def fit_fast(p,y,v,domains,reg=.005,passes=2):
    # Same coordinate objective/grid/tie tolerance as the accepted calibration.
    # Only the changed column and per-image counts need recomputation.
    t=np.full(40,.5,np.float32);groups=[]
    for d in sorted(set(domains)):
        ix=domains==d;truth=y[ix]==1;mask=v[ix].astype(bool);pred=p[ix]>.5
        pos=(truth&mask).sum(0);neg=(~truth&mask).sum(0);both=(pos>0)&(neg>0)
        tp=(pred&truth&mask).sum(0);tn=(~pred&~truth&mask).sum(0)
        groups.append(dict(p=p[ix],y=truth,v=mask,pred=pred,pos=pos,neg=neg,both=both,
                           ma=(div(tp,pos)+div(tn,neg))/2,
                           tp=(pred&truth&mask).sum(1),pc=(pred&mask).sum(1),gc=(truth&mask).sum(1)))
    def objective(j=None,candidate=None):
        scores=[]
        for g in groups:
            ma=g['ma'];a=g['tp'];b=g['pc']
            if j is not None:
                new=g['p'][:,j]>candidate
                dp=(new.astype(np.int32)-g['pred'][:,j].astype(np.int32))*g['v'][:,j]
                a=a+dp*g['y'][:,j];b=b+dp
                maj=(div((new&g['y'][:,j]&g['v'][:,j]).sum(),g['pos'][j])+div((~new&~g['y'][:,j]&g['v'][:,j]).sum(),g['neg'][j]))/2
                total=ma[g['both']].sum()+(maj-ma[j] if g['both'][j] else 0.)
            else:total=ma[g['both']].sum()
            m=float(total/max(1,g['both'].sum()))
            scores.append(harmonic(m,harmonic(float(div(a,b).mean()),float(div(a,g['gc']).mean()))))
        penalty=float(np.abs(t-.5).sum())
        if j is not None:penalty+=float(abs(candidate-.5)-abs(t[j]-.5))
        return float(np.mean(scores))-reg*penalty/40
    best=objective()
    for _ in range(passes):
        changed=False
        for j in range(40):
            if not v[:,j].any():continue
            original=t[j];winner=original
            for candidate in GRID:
                value=objective(j,candidate)
                if value>best+1e-8:best=value;winner=candidate;changed=True
            if winner!=original:
                for g in groups:
                    new=g['p'][:,j]>winner
                    dp=(new.astype(np.int32)-g['pred'][:,j].astype(np.int32))*g['v'][:,j]
                    g['tp']+=dp*g['y'][:,j];g['pc']+=dp;g['pred'][:,j]=new
                    g['ma'][j]=(div((new&g['y'][:,j]&g['v'][:,j]).sum(),g['pos'][j])+div((~new&~g['y'][:,j]&g['v'][:,j]).sum(),g['neg'][j]))/2
                t[j]=winner
        if not changed:break
    return t

def sensitivity(p,y,v,d,t):
    legacy=macro_score(p,y,v,d.tolist(),t);groups={}
    for name in sorted(set(d)):
        ix=d==name;a=p[ix]>t;b=y[ix]==1;m=v[ix].astype(bool)
        tp=(a&b&m).sum(0);tn=(~a&~b&m).sum(0);pos=(b&m).sum(0);neg=(~b&m).sum(0)
        ma=(div(tp,pos)+div(tn,neg))/2
        inter=(a&b&m).sum(1);pc=(a&m).sum(1);gc=(b&m).sum(1)
        f=float(div(2*inter,pc+gc).mean());g=dict(legacy['domains'][name]);g['Inst_F1']=f;g['Challenge_Avg']=harmonic(g['mA'],f)
        g['all_valid_safe_zero_mA']=float(ma[(pos+neg)>0].mean())
        g['zero_positive_attributes']=[ATTRIBUTES[j] for j in np.flatnonzero(pos==0)]
        groups[name]=g
    means={k:float(np.mean([g[k] for g in groups.values()])) for k in ['mA','Inst_F1','Challenge_Avg','all_valid_safe_zero_mA']}
    means['harmonic_of_macro_mA_and_mean_image_F1']=harmonic(means['mA'],means['Inst_F1'])
    means['harmonic_all_valid_safe_zero_mA_mean_image_F1']=harmonic(means['all_valid_safe_zero_mA'],means['Inst_F1'])
    return {'legacy':legacy,'mean_image_f1_sensitivity':{'macro':means,'domains':groups}}

def attribute_rows(p,y,v,d,t,label):
    rows=[]
    for domain in sorted(set(d)):
        ix=d==domain;pred=p[ix]>t;true=y[ix]==1;mask=v[ix].astype(bool)
        tp=(pred&true&mask).sum(0);tn=(~pred&~true&mask).sum(0);fp=(pred&~true&mask).sum(0);fn=(~pred&true&mask).sum(0)
        for j,a in enumerate(ATTRIBUTES):
            pos=tp[j]+fn[j];neg=tn[j]+fp[j]
            rows.append(dict(setting=label,domain=str(domain),attribute=a,threshold=float(t[j]),tp=int(tp[j]),tn=int(tn[j]),fp=int(fp[j]),fn=int(fn[j]),
                positives=int(pos),negatives=int(neg),precision=float(div(tp[j],tp[j]+fp[j])),recall=float(div(tp[j],pos)),
                f1=float(div(2*tp[j],2*tp[j]+fp[j]+fn[j])),mA=None if not(pos and neg) else float((tp[j]/pos+tn[j]/neg)/2),
                fpr=float(div(fp[j],neg)),fnr=float(div(fn[j],pos)),prevalence=float(div(pos,pos+neg)),predicted_prevalence=float(div(tp[j]+fp[j],pos+neg))))
    return rows

def main():
    start=time.perf_counter();root=Path(__file__).resolve().parents[1];out=root.parent/'diagnosis_1003877';out.mkdir(exist_ok=True)
    z=np.load(root.parent/'kaggle-v4-artifacts/runs/siglip2/best_val_predictions.npz')
    keep=z['calibration_eligible'].astype(bool);p=z['prob'][keep];y=z['y'][keep];v=z['valid'][keep];d=z['domains'][keep];paths=z['image_paths'][keep]
    cal=json.loads((root.parent/'kaggle-v4-artifacts/runs/siglip2/calibration.json').read_text());t=np.asarray(cal['thresholds'],np.float32);half=np.full(40,.5,np.float32)
    # Verify optimized sufficient-statistic fit against the untouched reference.
    small=np.concatenate([np.flatnonzero(d==name)[:60] for name in sorted(set(d))])
    a=fit_fast(p[small],y[small],v[small],d[small]);b=calibrate_thresholds(p[small],y[small],v[small],d[small].tolist())
    assert np.array_equal(a,b),(a,b)
    fitted=fit_fast(p,y,v,d);assert np.array_equal(fitted,t),(fitted,t)
    result={'calibration_optimizer_reference_parity':True,'rows':len(p),'excluded_rows':int((~keep).sum()),
            'raw':sensitivity(p,y,v,d,half),'accepted_calibration':sensitivity(p,y,v,d,t)}
    manifest=json.loads((root/'reports/data_mapped/image_manifest.json').read_text())
    pixel_groups=[manifest[str(name)]['pixel_sha256'] for name in paths]
    fold=np.asarray([int(hashlib.sha256(('2027:'+group).encode()).hexdigest()[:12],16)%5 for group in pixel_groups])
    folds=[];oof=np.zeros_like(y,dtype=bool);thresholds=[]
    for k in range(5):
        train=fold!=k;test=~train;th=fit_fast(p[train],y[train],v[train],d[train]);thresholds.append(th.tolist());oof[test]=p[test]>th
        folds.append({'fold':k,'fit_rows':int(train.sum()),'eval_rows':int(test.sum()),'domain_counts':{str(name):int(((d==name)&test).sum()) for name in sorted(set(d))},
                      'thresholds':th.tolist(),'raw':macro_score(p[test],y[test],v[test],d[test].tolist()),
                      'disjoint_calibrated':macro_score(p[test],y[test],v[test],d[test].tolist(),th)})
        print('Completed calibration fold',k,flush=True)
    result['five_fold_calibration']={'folds':folds,'pooled_oof':sensitivity(oof.astype(np.float32),y,v,d,half),
        'threshold_min':np.min(thresholds,axis=0).tolist(),'threshold_max':np.max(thresholds,axis=0).tolist(),
        'method':'seed2027 SHA256 decoded-pixel groups, five disjoint approximate domain-stratified folds; checkpoint had already used all validation for selection, so this isolates calibration optimism only'}
    transfer=[]
    for name in sorted(set(d)):
        source=d!=name;target=~source;th=fit_fast(p[source],y[source],v[source],d[source]);transfer.append({'calibration_held_domain':str(name),'source_rows':int(source.sum()),'evaluation_rows':int(target.sum()),'thresholds':th.tolist(),
          'raw':macro_score(p[target],y[target],v[target],d[target].tolist()),'transferred':macro_score(p[target],y[target],v[target],d[target].tolist(),th),
          'accepted_global_in_sample':macro_score(p[target],y[target],v[target],d[target].tolist(),t)})
        print('Completed calibration domain transfer',name,flush=True)
    result['domain_calibration_transfer']={'results':transfer,'limitation':'Existing model was trained on every dataset; this is threshold transfer, NOT leave-one-domain-out model evaluation.'}
    rows=attribute_rows(p,y,v,d,half,'raw')+attribute_rows(p,y,v,d,t,'accepted_calibrated')
    with (out/'per_attribute_known_domains.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for domain in sorted(set(d)):
        ix=d==domain
        summary[str(domain)]={}
        for label,threshold in [('raw',half),('accepted_calibrated',t)]:
            selected=[r for r in rows if r['domain']==domain and r['setting']==label]
            summary[str(domain)][label]={'metrics':macro_score(p[ix],y[ix],v[ix],d[ix].tolist(),threshold)['macro'],
              'FP':sum(r['fp'] for r in selected),'FN':sum(r['fn'] for r in selected),'TP':sum(r['tp'] for r in selected),'TN':sum(r['tn'] for r in selected),
              'true_positives_per_image':float((y[ix]*v[ix]).sum(1).mean()),'predicted_positives_per_image':float(((p[ix]>threshold)*v[ix]).sum(1).mean())}
    result['domain_confusions']=summary
    groups={'age':list(range(3)),'gender':[3],'hair':[4,5,6],'upper_length':[7],'upper_colors':list(range(8,20)),
            'lower_length':[20],'lower_colors':list(range(21,33)),'lower_type':[33,34],'accessories':list(range(35,40))}
    result['attribute_groups']={}
    for group,indices in groups.items():
        attrs={ATTRIBUTES[i] for i in indices};g={}
        for setting in ['raw','accepted_calibrated']:
            rs=[r for r in rows if r['setting']==setting and r['attribute'] in attrs]
            g[setting]={'mA':float(np.mean([np.mean([r['mA'] for r in rs if r['domain']==domain and r['mA'] is not None]) for domain in sorted(set(d))])),
                        'label_F1':float(np.mean([r['f1'] for r in rs]))}
        result['attribute_groups'][group]=g
    prevalence=json.loads((root.parent/'kaggle-v4-artifacts/runs/siglip2/prevalence.json').read_text())
    pp=np.asarray([prevalence[a] for a in ATTRIBUTES]);ratio=np.exp(1-2*pp)
    result['loss_threshold_analysis']={'lower_than_half':int((t<.5).sum()),'higher_than_half':int((t>.5).sum()),'at_grid_lower_bound':int(np.isclose(t,.22).sum()),
        'positive_negative_weight_ratio':ratio.tolist(),'effective_unweighted_posterior_threshold':(1/(1+np.exp(-(np.log(t/(1-t))-(1-2*pp))))).tolist(),
        'interpretation':'For an ideal weighted-BCE optimum under the sampled training distribution only; not a calibrated posterior guarantee or causal ablation.'}
    result['elapsed_cpu_seconds']=time.perf_counter()-start
    (out/'validation_generalization_analysis.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'calibration_folds.npz',image_paths=paths,fold=fold,decoded_pixel_group=np.asarray(pixel_groups))
    print(json.dumps({'cpu_seconds':result['elapsed_cpu_seconds'],'raw':result['raw']['legacy']['macro'],'calibrated':result['accepted_calibration']['legacy']['macro'],'oof':result['five_fold_calibration']['pooled_oof']['legacy']['macro']},indent=2),flush=True)

if __name__=='__main__':main()
