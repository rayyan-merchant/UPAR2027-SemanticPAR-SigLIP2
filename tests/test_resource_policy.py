import importlib.util,sys
from pathlib import Path
import pytest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'kaggle'))
from resource_policy import decide,next_fold,committed_history

def inputs():
    expected={h:{'partitions':{k:{'rows':n} for k,n in [('source_train',20000),('source_selection',1000),('source_calibration',1000),('sealed_target_evaluation',1000)]}} for h in ['PA100k','PETA','Market1501']}
    baseline={h:{'images_per_second':140} for h in expected}
    return expected,baseline

def test_parallel_uses_measured_resource_advantage():
    e,b=inputs();j={h:{'images_per_second':130} for h in ['PA100k','PETA']}
    r=decide(e,b,j,True,.1,14)
    assert r['allowed'] and r['mode']=='parallel' and r['session_cap_hours']==6

def test_unstable_pair_is_sequential():
    e,b=inputs();r=decide(e,b,{},False,.1,14)
    assert r['mode']=='sequential'

def test_budget_reserves_final_training():
    e,b=inputs();r=decide(e,b,{},False,.1,8.25)
    assert not r['allowed'] and r['session_cap_hours']==.25

def test_market_waits_for_pa_even_if_peta_finishes_first():
    pending=['Market1501']
    assert next_fold('parallel','PETA',pending) is None and pending==['Market1501']
    assert next_fold('parallel','PA100k',pending)=='Market1501' and not pending

def test_sequential_keeps_all_folds():
    pending=['PETA','Market1501']
    assert next_fold('sequential','PA100k',pending)=='PETA'
    assert next_fold('sequential','PETA',pending)=='Market1501'

def test_interrupted_history_retains_only_complete_lines(tmp_path):
    p=tmp_path/'history.jsonl';p.write_text('{"epoch":1}\n{"epoch":2')
    assert committed_history(p)==[{'epoch':1}]
