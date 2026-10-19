"""Provisional UPAR metrics and global threshold selection.

Dataset-macro averaging follows published 2027 rules. Empty-label behavior
follows safe divisions; check against scorer if organizers release it.
"""
import numpy as np


def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros_like(np.asarray(num, dtype=np.float64)), where=np.asarray(den)!=0)


def score_one_domain(probabilities, target, valid, thresholds):
    pred=np.asarray(probabilities) > np.asarray(thresholds)[None]
    true=np.asarray(target)==1
    mask=np.asarray(valid,dtype=bool)
    if pred.shape != true.shape or pred.shape[1]!=40: raise ValueError('Expected N x 40 probabilities and labels')
    attr_ok = mask.any(axis=0)
    tp=((pred & true)&mask).sum(0)
    tn=((~pred & ~true)&mask).sum(0)
    fp=((pred & ~true)&mask).sum(0)
    fn=((~pred & true)&mask).sum(0)
    # Exclude labels marked invalid across the dataset, per competition protocol.
    has_pos=(tp+fn)>0
    has_neg=(tn+fp)>0
    both=attr_ok&has_pos&has_neg
    ma_by_label = (_safe_div(tp,tp+fn) + _safe_div(tn,tn+fp))/2
    mA=float(ma_by_label[both].mean()) if both.any() else 0.0
    label_f1=_safe_div(2*tp,2*tp+fp+fn)
    label_f1=float(label_f1[attr_ok].mean()) if attr_ok.any() else 0.0
    pt=(pred & mask).sum(axis=1)
    gt=(true & mask).sum(axis=1)
    inter=(pred & true & mask).sum(axis=1)
    union=pt+gt-inter
    prec=float(_safe_div(inter,pt).mean())
    rec=float(_safe_div(inter,gt).mean())
    inst_f1=float(2*prec*rec/(prec+rec)) if prec+rec else 0.0
    inst_acc=float(_safe_div(inter,union).mean())
    avg=float(2*mA*inst_f1/(mA+inst_f1)) if mA+inst_f1 else 0.0
    return {'Challenge_Avg':avg,'mA':mA,'Label_F1':label_f1,
            'Inst_Acc':inst_acc,'Inst_Prec':prec,'Inst_Rec':rec,'Inst_F1':inst_f1}


def macro_score(probabilities, targets, valid, domains, thresholds=None):
    thresholds=np.asarray(thresholds if thresholds is not None else np.full(40,0.5),dtype=np.float32)
    p=np.asarray(probabilities); y=np.asarray(targets); v=np.asarray(valid); d=np.asarray(domains)
    groups={}
    for domain in sorted(set(domains)):
        ix=d==domain
        groups[str(domain)]=score_one_domain(p[ix],y[ix],v[ix],thresholds)
    means={key:float(np.mean([z[key] for z in groups.values()])) for key in next(iter(groups.values()))}
    return {'macro':means,'domains':groups}


def attribute_report(probabilities, target, valid, thresholds=None):
    """Per-attribute valid-label confusion counts, precision, recall and F1."""
    p=np.asarray(probabilities);y=np.asarray(target)==1;v=np.asarray(valid,bool)
    threshold=np.full(40,.5) if thresholds is None else np.asarray(thresholds)
    pred=p>threshold
    tp=(pred & y & v).sum(0);tn=(~pred & ~y & v).sum(0)
    fp=(pred & ~y & v).sum(0);fn=(~pred & y & v).sum(0)
    return {k:np.asarray(z).tolist() for k,z in {
        'tp':tp,'tn':tn,'fp':fp,'fn':fn,'precision':_safe_div(tp,tp+fp),
        'recall':_safe_div(tp,tp+fn),'f1':_safe_div(2*tp,2*tp+fp+fn)}.items()}


def calibrate_thresholds(probs,targets,valid,domains,base=0.5,reg=0.005,passes=2):
    """One global threshold per attribute, optimized on held-out validation only.

    Penalty discourages large shifts from 0.5, reducing domain overfit.
    """
    t=np.full(40,base,dtype=np.float32)
    candidates=np.asarray([0.22,0.28,0.34,0.40,0.45,0.50,0.55,0.60,0.66,0.72,0.78],np.float32)
    def objective(z):
        metric=macro_score(probs,targets,valid,domains,z)['macro']['Challenge_Avg']
        return metric-reg*np.mean(np.abs(z-.5))
    best=objective(t)
    for _ in range(passes):
        changed=False
        for j in range(40):
            if not (np.asarray(valid)[:,j]).any(): continue
            original=t[j]
            winner=original
            for candidate in candidates:
                t[j]=candidate
                score=objective(t)
                if score>best+1e-8:
                    best=score;winner=candidate;changed=True
            t[j]=winner
        if not changed: break
    return t
