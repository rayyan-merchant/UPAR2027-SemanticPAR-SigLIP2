"""Archive-once three-fold scheduler; independent GPUs, fixed source-only protocol."""
import gc,hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
FOLDS=['PA100k','PETA','Market1501']
NATIVE='55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832'

def env_for(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',UPAR_TORCH_THREADS='2',TOKENIZERS_PARALLELISM='false')

def snapshot():
    import psutil
    try:
        g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    except Exception as e:g=type(e).__name__
    return {'unix':time.time(),'cpu_percent':psutil.cpu_percent(),'ram_available_bytes':psutil.virtual_memory().available,'gpu':g}

def stop_all(active):
    for state in active.values():
        p=state['p']
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+15
    while any(s['p'].poll() is None for s in active.values()) and time.monotonic()<deadline:time.sleep(.5)
    for s in active.values():
        if s['p'].poll() is None:
            try:os.killpg(s['p'].pid,signal.SIGKILL)
            except ProcessLookupError:pass
        s['log'].close()

def main():
    base=Path('/kaggle/working');code=base/'lodo_code';suite=base/'lodo'
    os.chdir(code);sys.path[:0]=[str(code),str(code/'kaggle')];suite.mkdir(exist_ok=True)
    from upar.lodo import write_json,digest
    started=time.monotonic();active={};completed=[]
    def status(phase,**kw):
        data={'phase':phase,'completed_folds':completed,'new_competition_submissions':0,**kw}
        write_json(base/'STATUS.json',data);print(json.dumps(data),flush=True)
    try:
        status('preflight',training_started=False)
        subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','tokenizers>=0.22,<0.23','huggingface_hub>=0.34,<1','pytest','Pillow>=10','psutil'],check=True,timeout=150)
        import torch
        torch.set_num_threads(2)
        from upar.parity import code_digest
        if not torch.cuda.is_available():raise RuntimeError('GPU required; refusing CPU training')
        expected=json.loads((code/'reports/lodo/EXPECTED_SPLITS.json').read_text())
        coverage=json.loads((code/'reports/lodo/THREEFOLD_ARCHIVES.json').read_text())
        inputs=Path('/kaggle/input');original=[];recoveries=[]
        for current,dirs,files in os.walk(inputs):
            current=Path(current)
            if 'SUITE_RECEIPT.json' in files and current.name=='lodo':recoveries.append(current);dirs.clear();continue
            if 'foundation_native.pt' in files and 'PRETRAINED_PARITY.json' in files:original.append(current);dirs.clear()
            dirs[:]=[d for d in dirs if d not in {'runs','submission_candidate','hf-cache','upar_code','official_data'}]
        if len(original)!=1:raise RuntimeError('Require exactly one original general-purpose foundation')
        original=original[0]
        if len(recoveries)>1:raise RuntimeError('Ambiguous earlier LODO recovery source')
        if recoveries:
            shutil.copytree(recoveries[0],suite,dirs_exist_ok=True,ignore=shutil.ignore_patterns('staged_data','archive_cache','resource'))
        old_hw=json.loads((original.parent/'hardware.json').read_text())
        if torch.__version__!=old_hw['torch'] or torch.version.cuda!=old_hw['cuda']:raise RuntimeError('Exact historical GPU parity runtime mismatch')
        parity=json.loads((original/'PRETRAINED_PARITY.json').read_text())
        if not parity['passed'] or not parity['mixed_precision_checked'] or parity['code_sha256']!=code_digest():raise RuntimeError('Pretrained implementation parity missing/stale')
        if parity['prepared_sha256']!=NATIVE or digest(original/'foundation_native.pt')!=NATIVE:raise RuntimeError('Fresh foundation checksum mismatch')
        gates=suite/'gates';gates.mkdir(exist_ok=True)
        shutil.copyfile(original/'foundation_native.pt',gates/'foundation_native.pt');shutil.copyfile(original/'PRETRAINED_PARITY.json',gates/'PRETRAINED_PARITY.json')
        ck=torch.load(gates/'foundation_native.pt',map_location='cpu',weights_only=False,mmap=True)
        from upar.attributes import ATTRIBUTES
        if ck.get('trained',False) or ck['attributes']!=ATTRIBUTES:raise RuntimeError('PAR-trained initialization prohibited')
        del ck;gc.collect()
        evidence=suite/'original_evidence/data';evidence.mkdir(parents=True,exist_ok=True)
        audit=json.loads((original/'data/data_audit.json').read_text())
        if not audit['passed'] or audit['leakage_review_required']:raise RuntimeError('Prior full data audit not approved')
        shutil.copyfile(original/'data/data_audit.json',evidence/'data_audit.json')
        write_json(suite/'INITIALIZATION_RECEIPT.json',{'native_sha256':NATIVE,'parity_code_sha256':code_digest(),'all_domain_PAR_checkpoint_opened':False,'original_foundation_only':True,'independent_per_fold_process':True})
        hardware={'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'gpu_count':torch.cuda.device_count(),'cpus':os.cpu_count(),'free_disk_bytes':shutil.disk_usage(base).free}
        if hardware['free_disk_bytes']<8*1024**3:raise RuntimeError('Need8GiB working space for archive/data plus recoverable fold states')
        write_json(base/'hardware.json',hardware)
        (base/'requirements-executed.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        write_json(base/'source_hashes.json',{str(p.relative_to(code)):digest(p) for folder in ['upar','kaggle','scripts'] for p in (code/folder).glob('*.py')})
        # Tiny annotation files only; never enumerate a mounted image dataset.
        ann=list(inputs.rglob('annotations_train.csv'))
        if len(ann)!=1:raise RuntimeError('Ambiguous annotation index')
        index=suite/'annotation_index'
        for split in ['train','val']:
            source=ann[0].parent/('annotations_'+split+'.csv')
            if digest(source)!=expected['PA100k']['original_annotation_sha256'][split]:raise RuntimeError('Annotation bytes changed')
            target=index/'annotations/task1'/split/'gt.csv';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        from scripts.prepare_lodo_diagnosis import prepare
        for held in FOLDS:
            receipt=prepare(code,index,suite/held/'splits',held,manifest_path=original/'data/image_manifest.json')
            if receipt['partitions']!=expected[held]['partitions']:raise RuntimeError('Precommitted split changed')
        chosen=[]
        for item in coverage['archives']:
            candidates=list(inputs.rglob(item['archive']))
            if len(candidates)!=1:raise RuntimeError('Missing/ambiguous opaque archive')
            chosen.append(dict(item,path=str(candidates[0])))
        status('archive_staging',training_started=False,staging_limit_seconds=120)
        from archive_stage import stage
        staging=stage(chosen,suite/'staged_data',suite/'STAGING.json',120,30,120,0)
        if staging['images']!=coverage['coverage']['all_fold_union_images'] or staging['path_set_sha256']!=coverage['coverage']['required_path_set_sha256']:raise RuntimeError('Staged union differs from sealed manifests')
        from upar.data import load_rows
        os.environ['UPAR_TEST_IMAGE']=str(suite/'staged_data'/load_rows(suite/'PA100k/splits/source_train.csv')[0][0])
        subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests','--basetemp=/kaggle/working/pytest-fresh','--junitxml='+str(suite/'tests.xml')],check=True,timeout=150)
        bench=suite/'resource';bench.mkdir(exist_ok=True)
        def probes(specs,tag):
            jobs={};begin=time.monotonic();metrics=[];barrier=bench/(tag+'.go') if len(specs)>1 else None
            try:
                for held,gpu in specs:
                    output=bench/(tag+'_'+held+'.json');log=(bench/(tag+'_'+held+'.log')).open('w')
                    cmd=[sys.executable,str(code/'kaggle/resource_probe.py'),'--held',held,'--output',str(output)]
                    if barrier:cmd+=['--barrier',str(barrier)]
                    p=subprocess.Popen(cmd,env=env_for(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs[held]={'p':p,'log':log,'output':output}
                while any(s['p'].poll() is None for s in jobs.values()):
                    if time.monotonic()-begin>180:raise TimeoutError('Source-only resource probe timeout')
                    if barrier and not barrier.exists() and all(Path(str(barrier)+'.'+h+'.ready').exists() for h,_ in specs):barrier.write_text('go')
                    metrics.append(snapshot());time.sleep(1)
                if any(s['p'].returncode for s in jobs.values()):raise RuntimeError('Source-only resource probe failed')
                return {held:json.loads(s['output'].read_text()) for held,s in jobs.items()}
            finally:
                stop_all(jobs);write_json(bench/(tag+'_SYSTEM.json'),metrics)
        status('source_throughput_gates',training_started=False)
        baseline={}
        gpu1=1 if torch.cuda.device_count()>1 else 0
        for held,gpu in [('PA100k',0),('PETA',gpu1),('Market1501',0)]:baseline.update(probes([(held,gpu)],'solo_'+held))
        parallel={};stable=False
        if torch.cuda.device_count()>1:
            try:
                parallel=probes([('PA100k',0),('PETA',1)],'concurrent')
                ratios=[parallel[h]['images_per_second']/baseline[h]['images_per_second'] for h in ['PA100k','PETA']]
                overlap=min(x['probe_end_unix'] for x in parallel.values())-max(x['probe_start_unix'] for x in parallel.values())
                stable=min(ratios)>=.7 and overlap>1
            except Exception as e:write_json(bench/'PARALLEL_FALLBACK.json',{'error':str(e),'action':'sequential','target_metrics_used':False})
        from resource_policy import decide,next_fold,committed_history
        plan=json.loads((code/'reports/lodo/RESOURCE_AUTHORIZATION.json').read_text())
        decision=decide(expected,baseline,parallel,stable,(time.monotonic()-started)/3600,plan['quota_remaining_hours'])
        mode=decision['mode'];budget=decision['session_cap_hours'];projected=decision['projected_session_hours']
        solo_cost=decision['solo_fold_projected_hours'];parallel_cost=decision['concurrent_fold_projected_hours']
        write_json(suite/'RESOURCE_DECISION.json',decision)
        if budget<=0 or projected>budget-.25:raise RuntimeError('RESOURCE_CONFLICT: measured projection threatens final-training reserve; no fold launched')
        status('training',resource_decision=decision,training_started=True)
        write_json(base/'TRAINING_START.json',{'training_started':True,'resource_gate_passed':True})
        # Deadline begins at session setup, not when first optimizer step finishes.
        deadline=started+budget*3600;pending=['Market1501'] if mode=='parallel' else FOLDS.copy()
        def launch(held,gpu):
            selected=(bench/('concurrent_'+held+'.json')) if mode=='parallel' and held in parallel else bench/('solo_'+held+'_'+held+'.json')
            log=(suite/held/'process.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,str(code/'kaggle/lodo_fold_worker.py'),'--held',held,'--feasibility',str(selected)],
                env=env_for(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[held]={'p':p,'log':log,'gpu':gpu,'start':time.monotonic()}
            print('LAUNCH '+json.dumps({'held':held,'gpu':gpu,'pid':p.pid}),flush=True)
        if mode=='parallel':launch('PA100k',0);launch('PETA',1)
        else:launch(pending.pop(0),0)
        last_status=0
        while active:
            if time.monotonic()>deadline-120:raise RuntimeError('RESOURCE_CONFLICT: session cap reached; epoch-boundary best/last retained, unfinished folds not claimed')
            for held,s in list(active.items()):
                if s['p'].poll() is not None:
                    s['log'].close()
                    if s['p'].returncode!=0:raise RuntimeError('Infrastructure/training failure in '+held+'; see independent logs, do not drop a fold')
                    result=suite/held/'EVALUATION.json'
                    if not result.exists():raise RuntimeError('Fold exited without completed six-epoch locked target evaluation')
                    completed.append(held);gpu=s['gpu'];del active[held]
                    write_json(suite/'SUITE_RECEIPT.json',{'completed_folds':completed,'all_folds_complete':len(completed)==3,'submitted':False})
                    following=next_fold(mode,held,pending)
                    if following:launch(following,gpu)
            if time.monotonic()-last_status>30:
                progress={};samples=snapshot()
                for held,s in active.items():
                    path=suite/held/'training/progress.json'
                    try:p=json.loads(path.read_text())
                    except (FileNotFoundError,json.JSONDecodeError):p={'optimizer_updates_attempted':0}
                    p.update({'gpu':s['gpu'],'wall_seconds':time.monotonic()-s['start'],'planned_fold_hours':(parallel_cost if held in parallel_cost and mode=='parallel' else solo_cost)[held]})
                    if p.get('optimizer_updates_attempted',0)>0:
                        p['eta_training_seconds_estimate']=p.get('elapsed_seconds',p['wall_seconds'])/p['optimizer_updates_attempted']*max(0,p.get('optimizer_updates_total',0)-p['optimizer_updates_attempted'])
                    progress[held]=p
                    last_change=max((q.stat().st_mtime for q in [path,suite/held/'training/history.jsonl',suite/held/'process.log'] if q.exists()),default=time.time())
                    if time.time()-last_change>600:raise RuntimeError('Infrastructure stall: no training/log progress for600s in '+held)
                    history=suite/held/'training/history.jsonl'
                    if history.exists():
                        rows=committed_history(history)
                        if len(rows)>=3:
                            mean=sum(x['epoch_seconds'] for x in rows[-2:])/2
                            p['remaining_epoch_hours_estimate']=(6-len(rows))*mean/3600
                            if time.monotonic()+p['remaining_epoch_hours_estimate']*3600+300>deadline:raise RuntimeError('RESOURCE_CONFLICT: actual source trajectory exceeds remaining budget')
                status('training',active=progress,pending=pending,resource=samples,session_hours_elapsed=(time.monotonic()-started)/3600)
                last_status=time.monotonic()
            time.sleep(2)
        if set(completed)!=set(FOLDS):raise RuntimeError('Missing required fold')
        results=[json.loads((suite/h/'EVALUATION.json').read_text()) for h in FOLDS]
        write_json(suite/'CONSOLIDATED.json',{'folds':results,'elapsed_wall_seconds':time.monotonic()-started,
            'macro_target_raw':sum(r['target_raw']['legacy']['macro']['Challenge_Avg'] for r in results)/3,
            'macro_target_calibrated':sum(r['target_calibrated']['legacy']['macro']['Challenge_Avg'] for r in results)/3,
            'worst_target_raw':min(r['target_raw']['legacy']['macro']['Challenge_Avg'] for r in results),
            'worst_target_calibrated':min(r['target_calibrated']['legacy']['macro']['Challenge_Avg'] for r in results),'new_competition_submissions':0})
        status('complete',training_started=True)
        shutil.rmtree(suite/'staged_data');shutil.rmtree(suite/'archive_cache')
    except BaseException as e:
        stop_all(active);status('failed',error=type(e).__name__+': '+str(e),training_started=bool((base/'TRAINING_START.json').exists()),unfinished_folds=[h for h in FOLDS if h not in completed])
        # Derived local images are never recovery state; keep weights/logs/locks only.
        for p in [suite/'staged_data',suite/'archive_cache']:
            if p.exists() and p.resolve().is_relative_to(suite.resolve()):shutil.rmtree(p)
        raise

if __name__=='__main__':main()
