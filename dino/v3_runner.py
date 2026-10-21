"""One bounded frozen-DINO diagnostic: archive once, gate, cache, three heads."""
import gc,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from dino.v3_experiment import BASE,SUITE,WEIGHTS,Images,load_backbone,digest,write

FOLDS=['PA100k','PETA','Market1501'];CODE=Path(__file__).resolve().parents[1]

def env(gpu):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')

def find(inputs,name):
    found=[]
    for folder,dirs,files in os.walk(inputs):
        dirs[:]=[d for d in dirs if d not in {'staged_data','archive_cache','official_data','hf-cache','runs','submission_candidate','upar_code'}]
        if name in files:found.append(Path(folder)/name)
    return found

def jobs(specs,timeout,progress_paths,followups=None):
    active={};begin=time.monotonic();followups=followups or {}
    def launch(tag,args,gpu):
        log=(SUITE/(tag+'.log')).open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-m','dino.v3_experiment',*args],cwd=CODE,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[tag]=(p,log,time.monotonic())
    try:
        if timeout<300:raise RuntimeError('Insufficient remaining diagnostic budget for phase')
        for tag,args,gpu in specs:launch(tag,args,gpu)
        while active:
            if time.monotonic()-begin>timeout:raise TimeoutError('Diagnostic phase wall-time exceeded')
            for tag,(p,log,job_start) in list(active.items()):
                if p.poll() is not None:
                    log.close();del active[tag]
                    if p.returncode:raise RuntimeError('Worker failed: '+tag)
                    if tag in followups:launch(*followups.pop(tag))
                else:
                    progress=progress_paths.get(tag)
                    if progress and not progress.exists() and time.monotonic()-job_start>120:
                        raise TimeoutError('Worker has not produced first progress: '+tag)
                    if progress and progress.exists() and time.time()-progress.stat().st_mtime>180:
                        raise TimeoutError('Worker progress stalled: '+tag)
            time.sleep(1)
    finally:
        for p,log,_ in active.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            log.close()

def feasibility(paths):
    from dino.v3_api import pool_descriptors
    from torch.utils.data import DataLoader
    plan=json.loads((CODE/'reports/dino_v3/PLAN.json').read_text())
    for name,expected in plan['gpu_gate_sha256'].items():
        if digest(CODE/'reports/dino_v3'/name)!=expected:raise ValueError('Predeclared GPU gate evidence changed')
    hardware=json.loads((CODE/'reports/dino_v3/GATE_HARDWARE.json').read_text())
    parity=json.loads((CODE/'reports/dino_v3/GATE_PARITY.json').read_text())
    gate=json.loads((CODE/'reports/dino_v3/GATE_FEASIBILITY.json').read_text())
    if hardware['torch']!=torch.__version__ or hardware['cuda']!=torch.version.cuda:raise ValueError('GPU gate runtime differs')
    if not parity['passed'] or not gate['passed'] or parity['weights_sha256']!=plan['foundation_sha256']:raise ValueError('Verified GPU gate missing')
    native=load_backbone().cuda();loader=DataLoader(Images(paths*5),batch_size=64,num_workers=2,pin_memory=True,timeout=30)
    measured=0;begin=None;torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for k,(p,box,_) in enumerate(loader):
            if k==1:torch.cuda.synchronize();begin=time.monotonic()
            with torch.autocast('cuda',dtype=torch.float16):tokens=native(p.cuda(non_blocking=True))
            z=pool_descriptors(tokens,box.cuda());assert z.shape[1:]==(6,768)
            if not torch.isfinite(z).all():raise ValueError('Nonfinite real-image descriptors')
            if k>=1:measured+=len(p)
    torch.cuda.synchronize();seconds=time.monotonic()-begin;ips=measured/seconds
    peak=torch.cuda.max_memory_allocated()/1024**3;del native;gc.collect();torch.cuda.empty_cache()
    result={'passed':True,'source_only':True,'reused_complete_matching_GPU_parity':True,'images':measured,'seconds':seconds,'images_per_second':ips,'peak_allocated_GiB':peak,'feature_cache_projected_seconds_conservative':131076/ips,'feature_cache_bytes':131076*6*768*2,'target_version_cpu_parity_preverified':True}
    write(SUITE/'FEASIBILITY.json',result)
    if ips<75 or peak>12 or 131076/ips>1800:raise RuntimeError('Resource gate: feature extraction too slow/large')
    return result


def main():
    begin=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;SUITE.mkdir(exist_ok=True);inputs=Path('/kaggle/input')
    def status(phase,**kw):write(BASE/'STATUS.json',{'phase':phase,'new_competition_submissions':0,**kw});print(phase,json.dumps(kw),flush=True)
    try:
        status('preflight',training_started=False)
        plan=json.loads((CODE/'reports/dino_v3/PLAN.json').read_text())
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Two T4 GPUs required for bounded extraction')
        if shutil.disk_usage(BASE).free<8*1024**3:raise RuntimeError('Need8GiB free working space')
        write(BASE/'hardware.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'cpu_count':os.cpu_count(),'free_disk':shutil.disk_usage(BASE).free})
        (BASE/'requirements-executed.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        write(BASE/'source_hashes.json',{str(p.relative_to(CODE)):digest(p) for folder in ['dino','upar','scripts','kaggle'] for p in (CODE/folder).glob('*.py')})
        WEIGHTS.mkdir(exist_ok=True)
        model_files=[p for p in find(inputs,'RECEIPT.json') if json.loads(p.read_text()).get('repo')=='timm/vit_base_patch16_dinov3.lvd1689m']
        if len(model_files)!=1:raise ValueError('Require one pinned public timm DINOv3 dataset')
        source=model_files[0].parent
        for name in ['model.safetensors','config.json','DINOV3_LICENSE.txt','TIMM_APACHE_2_0.txt','RECEIPT.json','NATIVE_CPU_PARITY.json','TARGET_RUNTIME_PROBE.json']:
            shutil.copyfile(source/name,WEIGHTS/name)
        if digest(WEIGHTS/'model.safetensors')!=plan['foundation_sha256']:raise ValueError('Official pretrained weights hash changed')
        cpu=json.loads((WEIGHTS/'NATIVE_CPU_PARITY.json').read_text())
        if not cpu['passed'] or cpu['checkpoint_sha256']!=plan['foundation_sha256']:raise ValueError('CPU/checkpoint gate failed')
        target=json.loads((WEIGHTS/'TARGET_RUNTIME_PROBE.json').read_text())
        if not target['passed'] or target['torch']!='2.4.1+cpu' or target['timm_imported'] or target['transformers_imported']:raise ValueError('Native deployment-version CPU gate failed')
        if digest(CODE/'dino/v3_model.py')!=cpu['native_code_sha256']:raise ValueError('Native model changed since CPU parity')
        if digest(CODE/'dino/model.py')!=json.loads((CODE/'reports/dino_v3/GATE_BUILD_RECEIPT.json').read_text())['source_sha256']['dino/model.py']:raise ValueError('Six-descriptor/head implementation changed since GPU parity')
        write(SUITE/'INITIALIZATION_RECEIPT.json',{'pretrained_only':True,'PAR_weights_loaded':False,'foundation_sha256':plan['foundation_sha256'],'native_code_sha256':cpu['native_code_sha256'],'gpu_parity_sha256':plan['gpu_gate_sha256']['GATE_PARITY.json'],'all_folds_fresh_independent_heads':True})
        baseline=[p.parent for p in find(inputs,'CONSOLIDATED.json') if p.parent.name=='lodo']
        if len(baseline)!=1:raise ValueError('Exactly one original SigLIP LODO result source required')
        baseline=baseline[0]
        for held in FOLDS:
            fold=SUITE/held;fold.mkdir(exist_ok=True);shutil.copytree(baseline/held/'splits',fold/'splits',dirs_exist_ok=True)
            for name,info in plan['expected_splits'][held]['partitions'].items():
                if digest(fold/'splits'/(name+'.csv'))!=info['sha256']:raise ValueError('Sealed split differs')
            for name in ['target_predictions.npz','source_calibration_predictions.npz']:
                target=SUITE/'siglip'/held/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(baseline/held/name,target)
                if digest(target)!=plan['baseline_predictions'][held][name]:raise ValueError('SigLIP predictions differ from baseline lock')
        coverage=json.loads((CODE/'reports/lodo/THREEFOLD_ARCHIVES.json').read_text());chosen=[]
        for item in coverage['archives']:
            sources=find(inputs,item['archive'])
            if len(sources)!=1:raise ValueError('Missing opaque archive')
            chosen.append(dict(item,path=str(sources[0])))
        status('archive_staging',training_started=False)
        sys.path.insert(0,str(CODE/'kaggle'));from archive_stage import stage
        staging=stage(chosen,SUITE/'staged_data',SUITE/'STAGING.json',120,30,120,0)
        if staging['images']!=131076 or staging['path_set_sha256']!=coverage['coverage']['required_path_set_sha256']:raise ValueError('Archive path union differs')
        # Preserve original image hashes without re-reading mounted image files.
        manifests=find(inputs,'image_manifest.json')
        if not manifests:raise ValueError('Original audited image-hash manifest unavailable')
        manifest_path=next((p for p in manifests if digest(p)==plan['image_manifest_sha256']),None)
        if manifest_path is None:raise ValueError('Original image manifest checksum differs')
        manifest=json.loads(manifest_path.read_text())
        annotations=find(inputs,'annotations_train.csv')
        if len(annotations)!=1:raise ValueError('One official annotation index required')
        from upar.data import load_rows
        rows=[]
        for split in ['train','val']:
            path=annotations[0].parent/('annotations_'+split+'.csv')
            if digest(path)!=plan['expected_splits']['PA100k']['original_annotation_sha256'][split]:raise ValueError('Annotation bytes changed')
            rows+=load_rows(path)
        paths=[r[0] for r in rows]
        if len(paths)!=131076 or len(set(paths))!=131076:raise ValueError('Cache row identity is not one-to-one')
        write(SUITE/'INDEX.json',{'paths':paths,'domain_ids':[r[3] for r in rows],'original_sample_identity':paths,'image_sha256':[manifest[p]['sha256'] for p in paths],'audited_pixel_sha256':[manifest[p]['pixel_sha256'] for p in paths],'hidden_test_used':False})
        np.savez_compressed(SUITE/'labels.npz',y=np.stack([r[1] for r in rows]),valid=np.stack([r[2] for r in rows]),domains=np.array([r[3] for r in rows]))
        source_rows=load_rows(SUITE/'PA100k/splits/source_train.csv');by_domain={}
        for r in source_rows:by_domain.setdefault(r[3],[]).append(r[0])
        source_paths=[by_domain[d][i] for i in range(32) for d in sorted(by_domain)]
        # Engineering samples are source-only; override image path list for benchmark.
        original_index=(SUITE/'INDEX.json').read_text();write(SUITE/'INDEX.json',{'paths':source_paths})
        status('source_gpu_feasibility',training_started=False);feasibility(source_paths)
        (SUITE/'INDEX.json').write_text(original_index)
        if time.monotonic()-begin>600:raise TimeoutError('Pretraining setup budget exceeded')
        write(BASE/'TRAINING_START.json',{'source_feasibility_passed':True,'frozen_feature_extraction_started':True,'all_domain_training_started':False})
        status('frozen_descriptor_extraction',training_started=False)
        cache=np.lib.format.open_memmap(SUITE/'features.npy',mode='w+',dtype=np.float16,shape=(131076,6,768));cache.flush();del cache
        middle=65538
        jobs([('extract0',['--extract','0',str(middle)],0),('extract1',['--extract',str(middle),'131076'],1)],1800,
             {'extract0':SUITE/'EXTRACT_0.json','extract1':SUITE/f'EXTRACT_{middle}.json'})
        for start in [0,middle]:
            if not json.loads((SUITE/f'EXTRACT_{start}.json').read_text())['completed']:raise ValueError('Partial feature cache')
        write(SUITE/'FEATURE_RECEIPT.json',{'complete':True,'shape':[131076,6,768],'dtype':'float16','sha256':digest(SUITE/'features.npy'),'index_sha256':digest(SUITE/'INDEX.json'),'label_sha256':digest(SUITE/'labels.npz'),'backbone_sha256':plan['foundation_sha256'],'native_code_sha256':cpu['native_code_sha256'],'no_global_target_statistics_fit':True})
        # No image files are needed by cached-head folds; retain feature cache privately.
        shutil.rmtree(SUITE/'staged_data');shutil.rmtree(SUITE/'archive_cache')
        status('source_only_frozen_heads',training_started=True,all_domain_training_started=False)
        jobs([('PA100k',['--head','PA100k'],0),('PETA',['--head','PETA'],1)],min(5400,6300-(time.monotonic()-begin)),
             {h:SUITE/h/'PROGRESS.json' for h in FOLDS}, {'PA100k':('Market1501',['--head','Market1501'],0)})
        results=[json.loads((SUITE/h/'EVALUATION.json').read_text()) for h in FOLDS]
        summary={}
        for kind in ['dino_raw','dino_calibrated','ensemble_raw','ensemble_calibrated']:
            values=[r['reports'][kind]['legacy']['macro']['Challenge_Avg'] for r in results]
            summary[kind]={'macro':sum(values)/3,'worst':min(values),'domains':dict(zip(FOLDS,values))}
        new=summary['ensemble_calibrated'];old=plan['baseline_calibrated']
        promote=new['macro']>=.8023 and new['worst']>=.7772 and all(new['domains'][h]-old[h]>=-.005 for h in FOLDS)
        write(SUITE/'CONSOLIDATED.json',{'summary':summary,'ensemble_promotion_criteria_met':promote,'baseline_calibrated':old,'elapsed_seconds':time.monotonic()-begin,'automatic_all_domain_training':False,'new_competition_submissions':0})
        status('complete',frozen_lodo_complete=True,ensemble_promotion_criteria_met=promote,all_domain_training_started=False)
    except BaseException as e:
        status('failed',error=type(e).__name__+': '+str(e),all_domain_training_started=False)
        raise

if __name__=='__main__':main()
