"""User-authorized final B, archive staging once and source-only execution."""
import csv,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from dino.final_b import BASE,SUITE,WEIGHTS,digest,write,HEAD_HASHES
from dino.finetune_runner import find
from dino.final_a import INDEX_SHA,LABEL_SHA,split_validation

CODE=Path(__file__).resolve().parents[1]
FOLD=SUITE/'all_domain'

def load_label_arrays(path):
    """NpzFile indexing decompresses whole members; materialize each exactly once."""
    with np.load(path) as archive:
        return {name:archive[name] for name in ['y','valid','domains']}

def locate_pinned_siglip_predictions(inputs,expected):
    """Search metadata including runs/, pruning only bulky image/cache trees."""
    matches=[]
    for root,dirs,files in os.walk(inputs):
        dirs[:]=[d for d in dirs if d not in {'staged_data','archive_cache','official_data','hf-cache','upar_code','submission_candidate'}]
        if 'best_val_predictions.npz' in files:
            path=Path(root)/'best_val_predictions.npz'
            if digest(path)==expected:matches.append(path)
    if not matches:raise ValueError('Pinned original SigLIP raw source predictions missing')
    return sorted(matches)[0]

def validate_annotation_alignment(index,labels,parsed,offset):
    stop=offset+len(parsed)
    if index['paths'][offset:stop]!=[r[0] for r in parsed]:raise ValueError('Original path alignment differs')
    if not np.array_equal(labels['y'][offset:stop],np.stack([r[1] for r in parsed]),equal_nan=True):raise ValueError('Original label alignment differs')
    if not np.array_equal(labels['valid'][offset:stop],np.stack([r[2] for r in parsed])):raise ValueError('Original mask alignment differs')
    if not np.array_equal(labels['domains'][offset:stop],np.array([r[3] for r in parsed])):raise ValueError('Original domain alignment differs')

def run_worker(probe=False):
    tag='probe' if probe else 'training'
    with (SUITE/(tag+'.log')).open('a',buffering=1) as log:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        proc=subprocess.Popen([sys.executable,'-m','dino.final_b',*(['--probe'] if probe else [])],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        try:
            while proc.poll() is None:
                if probe and time.monotonic()-start>180:raise TimeoutError('Backward feasibility stalled')
                if not probe:
                    path=FOLD/'PROGRESS.json'
                    age=time.time()-path.stat().st_mtime if path.exists() else time.monotonic()-start
                    if age>180:raise TimeoutError('Training produced no progress for180s')
                time.sleep(1)
            if proc.returncode:raise RuntimeError(tag+' worker failed; inspect separate log')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)

def main():
    begin=time.monotonic();SUITE.mkdir(exist_ok=True);FOLD.mkdir(exist_ok=True)
    def status(phase,**kw):
        write(BASE/'STATUS.json',{'candidate':'B final exploratory','phase':phase,'elapsed_seconds':time.monotonic()-begin,'new_competition_submissions':0,**kw})
        print(phase,json.dumps(kw),flush=True)
    try:
        status('preflight',training_started=False)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if shutil.disk_usage(BASE).free<8*1024**3:raise RuntimeError('Need8GiB working storage')
        write(BASE/'hardware.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'free_disk':shutil.disk_usage(BASE).free})
        (BASE/'requirements-executed.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        write(BASE/'source_hashes.json',{str(p.relative_to(CODE)):digest(p) for folder in ['dino','upar','scripts','kaggle'] for p in (CODE/folder).glob('*.py')})
        inputs=Path('/kaggle/input');plan=json.loads((CODE/'reports/dino_v3/PLAN.json').read_text())
        from dino.limited_model import choose_foundation_receipt
        matches=[p for p in find(inputs,'RECEIPT.json') if json.loads(p.read_text()).get('repo')=='timm/vit_base_patch16_dinov3.lvd1689m']
        foundation=choose_foundation_receipt(matches,plan['foundation_sha256']).parent;WEIGHTS.mkdir(exist_ok=True)
        for name in ['model.safetensors','config.json','DINOV3_LICENSE.txt','TIMM_APACHE_2_0.txt','RECEIPT.json','NATIVE_CPU_PARITY.json','TARGET_RUNTIME_PROBE.json']:
            shutil.copyfile(foundation/name,WEIGHTS/name)
        if digest(WEIGHTS/'model.safetensors')!=plan['foundation_sha256']:raise ValueError('Pretrained foundation hash mismatch')
        cpu=json.loads((WEIGHTS/'NATIVE_CPU_PARITY.json').read_text())
        if not cpu['passed'] or digest(CODE/'dino/v3_model.py')!=cpu['native_code_sha256']:raise ValueError('Native pretrained parity code changed')
        hardware=json.loads((CODE/'reports/dino_v3/GATE_HARDWARE.json').read_text())
        parity=json.loads((CODE/'reports/dino_v3/GATE_PARITY.json').read_text())
        if hardware['torch']!=torch.__version__ or hardware['cuda']!=torch.version.cuda or not parity['passed']:raise ValueError('Preverified GPU environment/parity mismatch')
        for name,sha in plan['gpu_gate_sha256'].items():
            if digest(CODE/'reports/dino_v3'/name)!=sha:raise ValueError('Parity evidence changed')
        if digest(CODE/'dino/model.py')!=json.loads((CODE/'reports/dino_v3/GATE_BUILD_RECEIPT.json').read_text())['source_sha256']['dino/model.py']:raise ValueError('Head/pooling implementation changed')
        sources=[p.parent for p in find(inputs,'CONFIG.json') if p.parent.name=='candidate_a' and json.loads(p.read_text()).get('candidate')=='A']
        if len(sources)!=1:raise ValueError('Unique preserved Candidate A output required')
        a=sources[0]
        pinned={'best.pt':HEAD_HASHES['all_domain'],'CONFIG.json':'53867f902fe7eef2d954235e831954e1547f14230c066a522fc9febfbdebb570','PARTITIONS.npz':'505abd0df1f62c222da7ae5b1f051a12f79fcdc57ad76769532180dda2a45d2d','source_selection_MANIFEST.json':'8fea50be75aeb441f12344be24f30c58b8a7a770ce16f43dc40d9e7b01a1ca4b','source_calibration_MANIFEST.json':'53aa464a6848f64632c60f028b3e91aee46338e09d49c42af6be0a6024f78dc8','INDEX.json':INDEX_SHA,'labels.npz':LABEL_SHA}
        for name,sha in pinned.items():
            if digest(a/name)!=sha:raise ValueError('Candidate A provenance mismatch '+name)
        shutil.copyfile(a/'best.pt',FOLD/'INIT_HEAD.pt')
        for name in ['INDEX.json','labels.npz','PARTITIONS.npz']:
            shutil.copyfile(a/name,SUITE/name)
        status('auditing_source_metadata',training_started=False)
        index=json.loads((SUITE/'INDEX.json').read_text());labels=load_label_arrays(SUITE/'labels.npz');parts=np.load(SUITE/'PARTITIONS.npz')
        if len(index['paths'])!=131076 or len(set(index['paths']))!=131076:raise ValueError('Annotation index identity differs')
        selection,cal=split_validation(index)
        if not np.array_equal(parts['source_train'],np.arange(97669)) or not np.array_equal(parts['source_selection'],selection) or not np.array_equal(parts['source_calibration'],cal):raise ValueError('A source partitions changed')
        annotations=find(inputs,'annotations_train.csv')
        if len(annotations)!=1:raise ValueError('Unique official annotations required')
        rows=[];header=None
        from upar.data import load_rows
        from upar.attributes import ATTRIBUTES
        for name in ['train','val']:
            path=annotations[0].parent/('annotations_'+name+'.csv')
            if digest(path)!=plan['expected_splits']['PA100k']['original_annotation_sha256'][name]:raise ValueError('Official CSV changed')
            parsed=load_rows(path);offset=len(rows)
            validate_annotation_alignment(index,labels,parsed,offset)
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.reader(f);h=next(reader)
                if header is None:header=h
                if h!=header:raise ValueError('Attribute headers differ')
                rows.extend(row for row in reader if row)
        split=FOLD/'splits';split.mkdir(exist_ok=True)
        for name in ['source_train','source_selection','source_calibration']:
            with (split/(name+'.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(header);w.writerows(rows[int(i)] for i in parts[name])
        if set(index['audited_pixel_sha256'][i] for i in selection)&set(index['audited_pixel_sha256'][i] for i in cal):raise ValueError('Pixel split overlap')
        expected='8c7e126392b32dee9dbf6119f60ff55ac10fa0c2afc176f43e42c5df72d2ff65'
        predictions=locate_pinned_siglip_predictions(inputs,expected)
        write(SUITE/'SIGLIP_PREDICTION_DISCOVERY.json',{'file':str(predictions),'sha256':expected,'runs_directory_included':True})
        shutil.copyfile(predictions,SUITE/'siglip_validation_predictions.npz')
        write(SUITE/'INITIALIZATION_RECEIPT.json',{'foundation_sha256':plan['foundation_sha256'],'A_head_sha256':HEAD_HASHES['all_domain'],'A_head_epoch':30,'all_domain_PAR_backbone_loaded':False,'source_training_rows':97669,'source_selection_rows':len(selection),'source_calibration_rows':len(cal),'attributes':ATTRIBUTES,'A_partition_npz_sha256':pinned['PARTITIONS.npz'],'originals_modified':False})
        coverage=json.loads((CODE/'reports/lodo/THREEFOLD_ARCHIVES.json').read_text());chosen=[]
        for item in coverage['archives']:
            found=find(inputs,item['archive'])
            if len(found)!=1:raise ValueError('Archive missing or ambiguous')
            chosen.append(dict(item,path=str(found[0])))
        status('archive_staging',training_started=False)
        sys.path.insert(0,str(CODE/'kaggle'));from archive_stage import stage
        staging=stage(chosen,SUITE/'staged_data',SUITE/'STAGING.json',120,30,120,0)
        if staging['images']!=131076 or staging['path_set_sha256']!=coverage['coverage']['required_path_set_sha256']:raise ValueError('Archive path-set mismatch')
        status('source_backward_feasibility',training_started=False);run_worker(probe=True)
        probe=json.loads((SUITE/'all_domain_PROBE.json').read_text());ips=probe['images_per_second']
        projected=(8*97669/ips*1.35+(8*len(selection)+len(selection)+len(cal))/(ips*2)+600)/3600
        decision={'measured_backward_images_per_second':ips,'peak_allocated_GiB':probe['peak_allocated_GiB'],'projected_max8_epoch_hours':projected,'quota_remaining_at_launch_hours':7.58,'reserve_hours':2,'artificial_training_wall_time_cap':None,'allowed':projected<5.58,'source_only':True,'extension_rule':'six mandatory; epoch7/8 only if preceding epoch is source best by>=1e-4'}
        write(SUITE/'RESOURCE_DECISION.json',decision)
        if not decision['allowed']:raise RuntimeError('Quota reserve conflict; no training launched')
        if time.monotonic()-begin>600:raise TimeoutError('Pretraining setup exceeded600s')
        write(BASE/'TRAINING_START.json',{'backward_gate_passed':True,'all_domain_B_authorized':True,'no_arbitrary_training_time_cap':True})
        status('final_B_training',training_started=True);run_worker()
        # Extracted images and archive caches are transient; persistent checkpoints remain.
        for name in ['staged_data','archive_cache']:
            path=SUITE/name
            if path.exists():shutil.rmtree(path)
        status('complete',training_complete=True,offline_deployment_gates_pending=True)
    except BaseException as e:
        status('failed',error=type(e).__name__+': '+str(e));raise

if __name__=='__main__':main()
