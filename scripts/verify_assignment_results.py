"""Reproduce assignment tables from public aggregate evidence; no ML dependencies."""
import argparse,csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDS=['PA100k','PETA','Market1501']
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def build():
    rows=[]
    def add(name,kind,setting,metrics,source,decision,official=False):
        for domain,m in metrics.items():
            rows.append(dict(experiment=name,evaluation=kind,setting=setting,domain=domain,Challenge_Avg=m.get('Challenge_Avg'),mA=m.get('mA'),instance_F1=m.get('Inst_F1'),precision=m.get('Inst_Prec'),recall=m.get('Inst_Rec'),official_score=official,decision=decision,evidence=source))
        if kind=='LODO':
            fields=['Challenge_Avg','mA','Inst_F1','Inst_Prec','Inst_Rec']
            macro={k:sum(v[k] for v in metrics.values())/3 for k in fields}
            rows.append(dict(experiment=name,evaluation=kind,setting=setting,domain='MACRO',Challenge_Avg=macro['Challenge_Avg'],mA=macro['mA'],instance_F1=macro['Inst_F1'],precision=macro['Inst_Prec'],recall=macro['Inst_Rec'],official_score=False,decision=decision,evidence=source))
            rows.append(dict(experiment=name,evaluation=kind,setting=setting,domain='WORST',Challenge_Avg=min(v['Challenge_Avg'] for v in metrics.values()),mA=None,instance_F1=None,precision=None,recall=None,official_score=False,decision=decision,evidence=source))
    off=read('reports/CODABENCH_SUBMISSION_RECEIPT.json')
    require(off['submission_id']==1003877 and off['status']=='Finished','Missing completed original entry')
    require(off['sha256']=='76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308','Original ZIP identity changed')
    add('Original SigLIP2','OFFICIAL','production calibrated',{'OFFICIAL':dict(Challenge_Avg=off['score'],mA=off['mA'],Inst_F1=off['instance_f1'],Inst_Prec=off['instance_precision'],Inst_Rec=off['instance_recall'])},'reports/CODABENCH_SUBMISSION_RECEIPT.json','Evaluated fallback',True)
    for name,folder,decision in [('SigLIP2 baseline','reports/lodo/results','Locked baseline'),('Equal-domain sampling','reports/lodo/equal_domain/results','Rejected'),('Candidate C augmentation','reports/competitive_final/candidate_c','Rejected')]:
        for setting in ['raw','calibrated']:
            metrics={d:read(f'{folder}/{d}_EVALUATION.json')[f'target_{setting}']['legacy']['macro'] for d in FOLDS}
            add(name,'LODO',setting,metrics,folder,decision)
    frozen=read('reports/dino_v3/results/SUMMARY.json')
    for name,key,decision in [('Frozen DINOv3 standalone','dino','Rejected standalone'),('Candidate A frozen ensemble','ensemble','Failed promotion; separate exploratory package')]:
        for setting in ['raw','calibrated']:
            metrics={v['domain']:v for v in frozen['fold_metrics'] if v['setting']==f'{key}_{setting}'}
            require(set(metrics)==set(FOLDS),'Incomplete frozen folds')
            add(name,'LODO',setting,metrics,'reports/dino_v3/results/SUMMARY.json',decision)
    b=read('reports/competitive_final/candidate_b/SUMMARY.json')
    for name,key in [('Candidate B standalone','dino'),('Candidate B adapted ensemble','ensemble')]:
        for setting in ['raw','calibrated']:
            add(name,'LODO',setting,{d:b['domains'][d]['metrics'][f'{key}_{setting}'] for d in FOLDS},'reports/competitive_final/candidate_b/SUMMARY.json','Failed advancement; separate exploratory final run')
    a=read('reports/competitive_final/CANDIDATE_A_INDEPENDENT_VERIFICATION.json')
    add('Final Candidate A','SOURCE_SELECTION','calibrated ensemble',{'SOURCE_SELECTION':a['known_source_metrics']['ensemble_calibrated']['selection']},'reports/competitive_final/CANDIDATE_A_INDEPENDENT_VERIFICATION.json','Packaged; upload not confirmed')
    final=read('reports/competitive_final/final_b_results/INDEPENDENT_VERIFICATION.json')
    require(final['independent_audit_passed'] and final['epochs_completed']==8 and final['best_epoch']==8,'Final B training/audit incomplete')
    add('Final Candidate B','SOURCE_SELECTION','calibrated ensemble',{'SOURCE_SELECTION':final['metrics']['selection']['ensemble_calibrated']['macro']},'reports/competitive_final/final_b_results/INDEPENDENT_VERIFICATION.json','Prepared; official evaluation incomplete')
    require(not frozen['promoted'] and not b['advancement_gate_passed'],'Historical negative gates changed')
    equal=read('reports/lodo/equal_domain/results/COMPARISON.json')
    require(not equal['predeclared_criteria_met'],'Equal sampling decision changed')
    require(b['macro']-frozen['settings']['ensemble_calibrated']['Challenge_Avg']<.002,'B historical margin no longer fails')
    upload=read('reports/competitive_final/CANDIDATE_A_UPLOAD_RECEIPT.json')
    require(upload['submission_id'] is None and upload['official_score'] is None,'A official state changed; revise report')
    runtime=read('reports/competitive_final/final_b_results/runtime_v1/STATUS.json')
    progress=read('reports/competitive_final/final_b_results/runtime_v1/FULL_WORKLOAD.progress.json')
    require(runtime['phase']=='failed' and progress['images']==26624 and not runtime['A10_runtime_verified'],'Runtime result changed')
    for rel,digest in read('reports/lodo/equal_domain/BASELINE_LOCK.json')['files'].items():
        require(sha(ROOT/rel)==digest,'Locked baseline modified: '+rel)
    expected={('SigLIP2 baseline','MACRO'):.7972601615735578,('Equal-domain sampling','MACRO'):.7960893269758397,('Candidate A frozen ensemble','MACRO'):.800963783,('Candidate B adapted ensemble','MACRO'):.8025077541728867,('Candidate C augmentation','MACRO'):.7980759705878349}
    for (name,domain),score in expected.items():
        row=next(x for x in rows if x['experiment']==name and x['domain']==domain and x['setting']=='calibrated')
        require(math.isclose(row['Challenge_Avg'],score,abs_tol=1e-9),'Reported score mismatch: '+name)
    return rows
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true',help='Write refreshed tables; default verifies existing tables without changing files');a=parser.parse_args()
    rows=build();target=ROOT/'assignment/RESULTS.json'
    if a.write:
        target.write_text(json.dumps({'scope':'Retained aggregate evidence; not a new raw-prediction audit','records':rows},indent=2)+'\n',encoding='utf-8')
        with (ROOT/'assignment/RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    else:
        require(read('assignment/RESULTS.json')['records']==rows,'RESULTS.json is stale; inspect before --write')
        with (ROOT/'assignment/RESULTS.csv').open(encoding='utf-8',newline='') as f:csvrows=list(csv.DictReader(f))
        require(len(csvrows)==len(rows),'CSV row count changed')
        for got,expected in zip(csvrows,rows):
            require(got=={k:('' if v is None else str(v)) for k,v in expected.items()},'CSV content mismatch')
    print(json.dumps({'passed':True,'records':len(rows),'official_entries_confirmed':1,'original_submission_id':1003877,'baseline_hash_locks':'unchanged','new_training_started':False,'new_official_upload':False},indent=2))
if __name__=='__main__':main()
