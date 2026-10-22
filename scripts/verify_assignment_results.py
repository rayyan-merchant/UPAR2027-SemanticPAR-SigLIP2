"""Check published result-table arithmetic; never reads private research records."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDS={'PA100k','PETA','Market1501'}
FIELDS=['Challenge_Avg','mA','instance_F1','precision','recall']
def require(ok,message):
    if not ok:raise ValueError(message)
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-9)
def main():
    with (ROOT/'assignment/RESULTS.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==73,'Expected 73 published aggregate rows')
    for row in rows:
        for key in FIELDS:
            row[key]=float(row[key]) if row[key] else None
            if row[key] is not None:require(math.isfinite(row[key]) and 0<=row[key]<=1,'Invalid metric range')
    official=[x for x in rows if x['official_score']=='True']
    require(len(official)==1 and official[0]['submission_id']=='1003877','Only original 1003877 is confirmed official')
    require(close(official[0]['Challenge_Avg'],.770450),'Official Challenge_Avg changed')
    groups=defaultdict(list)
    for row in rows:
        if row['evaluation']=='LODO':groups[(row['experiment'],row['setting'])].append(row)
    require(len(groups)==14,'Incomplete LODO settings')
    for name,values in groups.items():
        fold=[v for v in values if v['domain'] in FOLDS]
        require(len(values)==5 and {v['domain'] for v in fold}==FOLDS,'Missing fold/macro/worst: '+str(name))
        macro=next(x for x in values if x['domain']=='MACRO');worst=next(x for x in values if x['domain']=='WORST')
        for key in FIELDS:require(close(macro[key],sum(x[key] for x in fold)/3),'Macro arithmetic mismatch: '+str(name))
        require(close(worst['Challenge_Avg'],min(x['Challenge_Avg'] for x in fold)),'Worst arithmetic mismatch')
        for x in fold:
            ma,f1=x['mA'],x['instance_F1'];h=2*ma*f1/(ma+f1) if ma+f1 else 0
            require(close(x['Challenge_Avg'],h),'Per-domain Challenge formula mismatch')
    def score(name,domain='MACRO'):
        return next(x['Challenge_Avg'] for x in groups[(name,'calibrated')] if x['domain']==domain)
    expected={'SigLIP2 baseline':.7972601615735578,'Equal-domain sampling':.7960893269758397,'Candidate A frozen ensemble':.800963783,'Candidate B adapted ensemble':.8025077541728867,'Candidate C augmentation':.7980759705878349}
    for name,value in expected.items():require(close(score(name),value),'Published macro changed: '+name)
    base=score('SigLIP2 baseline');a=score('Candidate A frozen ensemble');b=score('Candidate B adapted ensemble')
    require(score('Equal-domain sampling')<base,'Equal sampling rejection inconsistent')
    require(a<.8023 and b-a<.002,'Historical A/B advancement margins must remain failed')
    require(score('Candidate C augmentation','WORST')<score('SigLIP2 baseline','WORST'),'C worst-domain rejection inconsistent')
    for d in FOLDS:require(score('Candidate B adapted ensemble',d)>score('Candidate A frozen ensemble',d)>score('SigLIP2 baseline',d),'Expected per-domain ensemble gain missing')
    print(json.dumps({'passed':True,'published_rows_checked':len(rows),'LODO_settings_checked':len(groups),'official_entries_confirmed':1,'original_submission_id':1003877,'scope':'Published aggregate-table consistency, not a raw-prediction audit','new_training_started':False},indent=2))
if __name__=='__main__':main()
