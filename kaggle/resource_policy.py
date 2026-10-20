"""Source-throughput-only scheduling policy; no accuracy or target outcomes."""
import json
FOLDS=['PA100k','PETA','Market1501']
def fold_hours(partitions,images_per_second):
    if images_per_second<=0:raise ValueError('Positive throughput required')
    train=partitions['source_train']['rows'];selection=partitions['source_selection']['rows']
    post=partitions['source_calibration']['rows']+partitions['sealed_target_evaluation']['rows']
    return (6*train/images_per_second*1.5+(12*selection+post)/(images_per_second*1.25)+300)/3600

def decide(expected,baseline,parallel,stable,setup_hours,quota_remaining):
    solo={h:fold_hours(expected[h]['partitions'],baseline[h]['images_per_second']) for h in FOLDS}
    joint={h:fold_hours(expected[h]['partitions'],parallel[h]['images_per_second']) for h in parallel} if stable else {}
    penalty=min(parallel[h]['images_per_second']/baseline[h]['images_per_second'] for h in joint) if joint else 1.
    market=fold_hours(expected['Market1501']['partitions'],baseline['Market1501']['images_per_second']*penalty)
    sequential=sum(solo.values());concurrent=max(joint.get('PETA',999),joint.get('PA100k',999)+market)
    mode='parallel' if stable and concurrent<sequential else 'sequential'
    budget=min(6.,quota_remaining-8.);projection=(concurrent if mode=='parallel' else sequential)+setup_hours
    return {'mode':mode,'concurrent_stable':stable,'source_only':True,'solo_fold_projected_hours':solo,
        'concurrent_fold_projected_hours':joint,'projected_session_hours':projection,'session_cap_hours':budget,
        'final_training_reserve_hours':8,'quota_remaining_at_submission':quota_remaining,
        'allowed':budget>0 and projection<=budget-.25,
        'projection_limitations':'48 source batches;1.5x training margin, raw+EMA validation and300s/fold checkpoints. Market concurrent throughput conservatively adjusted by measured contention; not convergence evidence.'}

def next_fold(mode,finished,pending):
    if not pending:return None
    if mode=='parallel' and finished!='PA100k':return None
    return pending.pop(0)

def committed_history(path):
    # A live or interrupted writer may leave a trailing partial JSON line.
    return [json.loads(line) for line in path.read_text().split('\n')[:-1] if line]
