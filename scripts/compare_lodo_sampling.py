"""Verify one predeclared equal-domain comparison; never train or submit."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_lodo import summarize, table
from upar.lodo import digest
from upar.metrics import macro_score

FOLDS=['PA100k','PETA','Market1501']


def compare(artifacts):
    artifacts=Path(artifacts);public=ROOT/'reports/lodo/equal_domain/results'
    lock=json.loads((ROOT/'reports/lodo/equal_domain/BASELINE_LOCK.json').read_text())
    for name,expected in lock['files'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Locked baseline changed: '+name)
    # Recompute target metrics, exact source-only thresholds and CSV/lock hashes.
    summarize(artifacts,ROOT/'reports/lodo/equal_domain/INDEPENDENT_EVALUATION.md',public)
    baseline=[json.loads((ROOT/f'reports/lodo/results/{h}_EVALUATION.json').read_text()) for h in FOLDS]
    new=[json.loads((artifacts/h/'EVALUATION.json').read_text()) for h in FOLDS]
    configs=[];comparison=[];groups=[];exposures=[];sources=[]
    keys=['Challenge_Avg','mA','Inst_F1','Inst_Prec','Inst_Rec','Label_F1']
    for h,b,n in zip(FOLDS,baseline,new):
        bc=json.loads((ROOT/f'reports/lodo/results/{h}/FOLD_CONFIG.json').read_text())
        nc=json.loads((artifacts/h/'FOLD_CONFIG.json').read_text())
        extras={k:nc.pop(k) for k in ['domain_sampling_alpha','record_sampling_exposure','baseline_commit','development_comparison']}
        if nc!=bc or extras!={'domain_sampling_alpha':0.,'record_sampling_exposure':True,'baseline_commit':'4b60ffc','development_comparison':True}:
            raise ValueError('Unexpected methodological configuration change: '+h)
        for key,value in b['source_train_supervision'].items():
            if key!='expected_sampling_domain_mass' and n['source_train_supervision'][key]!=value:
                raise ValueError('Source frequencies/BCE/support changed: '+h+'/'+key)
        if any(v!=.5 for v in n['source_train_supervision']['expected_sampling_domain_mass'].values()):
            raise ValueError('Not equal-domain sampling')
        configs.append({'held_out':h,'only_methodological_change':'sampling alpha .5 -> 0','source_frequency_and_BCE_exact':True})
        histories=[json.loads(x) for x in (artifacts/h/'training/history.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in histories]!=list(range(1,7)):raise ValueError('Incomplete comparison fold')
        prev={d:0 for d in n['source_train_supervision']['domain_counts']}
        for row in histories:
            e=row['sampling_exposure'];count=n['source_train_supervision']['domain_counts']
            if set(e['draws'])!=set(count) or sum(e['draws'].values())!=sum(count.values()):raise ValueError('Exposure draws differ')
            for d,size in count.items():
                unique=e['unique_images_epoch'][d];cumulative=e['unique_images_cumulative'][d]
                if not 0<unique<=min(size,e['draws'][d]) or not max(prev[d],unique)<=cumulative<=min(size,prev[d]+unique):
                    raise ValueError('Invalid unique source image counts')
                if abs(e['fractions'][d]-.5)>.03:raise ValueError('Unexpected measured exposure')
                exposures.append([h,row['epoch'],d,e['draws'][d],e['fractions'][d],unique,cumulative,size,row['epoch_seconds']])
                prev[d]=cumulative
        # Independently reproduce source selection scores too.
        z=np.load(artifacts/h/'training/best_val_predictions.npz')
        calibration=json.loads((artifacts/h/'calibration.json').read_text())
        for setting,threshold in [('raw',np.full(40,.5)),('calibrated',calibration['thresholds'])]:
            actual=macro_score(z['prob'],z['y'],z['valid'],z['domains'].tolist(),threshold)['macro']
            if any(abs(actual[k]-n['source'][setting]['legacy']['macro'][k])>1e-10 for k in actual):raise ValueError('Source score mismatch')
            bm=b['target_'+setting]['legacy']['macro'];nm=n['target_'+setting]['legacy']['macro']
            comparison.append([h,setting]+[v for k in keys for v in [bm[k],nm[k],nm[k]-bm[k]]])
            sources.append([h,setting,b['source'][setting]['legacy']['macro']['Challenge_Avg'],n['source'][setting]['legacy']['macro']['Challenge_Avg']])
            for group,old in b['target_attribute_groups_'+setting].items():
                fresh=n['target_attribute_groups_'+setting][group]
                groups.append([h,setting,group,old['mA'],fresh['mA'],None if old['mA'] is None or fresh['mA'] is None else fresh['mA']-old['mA'],old['label_F1'],fresh['label_F1']])
    rawcal={}
    for setting in ['raw','calibrated']:
        a=[r['target_'+setting]['legacy']['macro']['Challenge_Avg'] for r in baseline]
        b=[r['target_'+setting]['legacy']['macro']['Challenge_Avg'] for r in new]
        rawcal[setting]={'baseline_macro':float(np.mean(a)),'new_macro':float(np.mean(b)),'macro_gain':float(np.mean(b)-np.mean(a)),
                         'baseline_worst':min(a),'new_worst':min(b),'worst_gain':min(b)-min(a),'domain_gains':dict(zip(FOLDS,[y-x for x,y in zip(a,b)]))}
    sense=[]
    for results in [baseline,new]:
        a=[r['target_calibrated']['mean_image_f1_sensitivity']['macro']['Challenge_Avg'] for r in results];sense.append((float(np.mean(a)),min(a)))
    c=json.loads((ROOT/'reports/lodo/equal_domain/RESOURCE_AUTHORIZATION.json').read_text())['adoption_criteria']
    r=rawcal['calibrated'];eligible=(r['macro_gain']>=c['minimum_macro_gain'] and r['worst_gain']>=c['minimum_worst_domain_gain']
        and min(r['domain_gains'].values())>=-c['maximum_any_domain_regression'] and sense[1][0]>sense[0][0] and sense[1][1]>sense[0][1])
    decision={'recommendation':'equal-domain sampling' if eligible else 'retain original square-root-domain sampling',
              'predeclared_criteria_met':eligible,'criteria':c,'summary':rawcal,'mean_image_f1_macro_worst':sense,'verified_controls':configs,
              'single_development_comparison_not_unbiased_test':True,'final_training_or_upload_authorized':False}
    public.mkdir(exist_ok=True,parents=True)
    (public/'COMPARISON.json').write_text(json.dumps(decision,indent=2)+'\n')
    for name,head,rows in [('METRIC_COMPARISON.csv',['held_out','setting']+[x for k in keys for x in [k+'_baseline',k+'_equal',k+'_delta']],comparison),
                           ('GROUP_COMPARISON.csv',['held_out','setting','group','baseline_mA','equal_mA','delta_mA','baseline_label_F1','equal_label_F1'],groups),
                           ('SAMPLING_EXPOSURE.csv',['held_out','epoch','source_domain','draws','draw_fraction','unique_images_epoch','unique_images_cumulative','available_source_images','epoch_seconds'],exposures)]:
        with (public/name).open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(head);w.writerows(rows)
    report='# Equal-domain sampling comparison — measured results\n\n'
    report+='Locked baseline4b60ffc. One predeclared development comparison; targets were previously inspected. No hidden leaderboard tuning or production replacement.\n\n'
    for setting in ['raw','calibrated']:
        rows=[]
        for h,b,n in zip(FOLDS,baseline,new):
            old=b['target_'+setting]['legacy']['macro'];fresh=n['target_'+setting]['legacy']['macro']
            rows.append([h]+[f'{v:.6f}' for v in [old['Challenge_Avg'],fresh['Challenge_Avg'],fresh['Challenge_Avg']-old['Challenge_Avg'],fresh['mA']-old['mA'],fresh['Inst_F1']-old['Inst_F1'],fresh['Inst_Prec']-old['Inst_Prec'],fresh['Inst_Rec']-old['Inst_Rec']]])
        report+='## '+setting.title()+' target changes\n\n'+table(['Target','Baseline','Equal','Delta score','Delta mA','Delta instance-F1','Delta precision','Delta recall'],rows)+'\n'
    report+='## Predeclared decision\n\n**'+decision['recommendation']+'**. Criteria met: '+str(eligible)+'.\n\n'
    report+='Macro/worst scores, all metrics and group effects: reports/lodo/equal_domain/results/COMPARISON.json, METRIC_COMPARISON.csv and GROUP_COMPARISON.csv. Actual per-domain draws/unique source images/epoch times: SAMPLING_EXPOSURE.csv. Baseline domain exposure is expected, not measured.\n\n'
    report+='This rule is a conservative development decision, not statistical proof; one seed, source selection, scorer ambiguity and incomplete convergence remain. Review actual group/trajectory changes before final delivery. No further training or competition upload was launched. Append measured runtime/quota and the deadline completion plan after review.\n'
    (ROOT/'LODO_SAMPLING_COMPARISON.md').write_text(report,encoding='utf-8')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);a=p.parse_args();compare(a.artifacts)
