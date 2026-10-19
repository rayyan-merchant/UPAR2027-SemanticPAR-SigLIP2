"""Kaggle CLI pipeline: data audit -> tests -> pretrained parity -> GPU benchmark -> one training run."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def run(*args):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True)


def main():
    base=Path('/kaggle/working')
    code=base/'upar_code';code.mkdir(exist_ok=True)
    # Bootstrap has already unpacked the embedded source into this directory.
    os.chdir(code);sys.path.insert(0,str(code))
    archives=list(Path('/kaggle/input').rglob('official_data.zip'))
    mounted_train=list(Path('/kaggle/input').glob('**/annotations/task1/train/gt.csv'))
    if len(archives)!=1 and len(mounted_train)!=1:
        raise RuntimeError('Official private data source is not attached; refusing setup/training')
    os.environ['HF_HOME']=str(base/'hf-cache')
    run('-m','pip','install','-q','transformers==4.57.1','tokenizers>=0.22,<0.23',
        'huggingface_hub>=0.34,<1','hf_xet','safetensors','sentencepiece','pytest','Pillow>=10')
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Kaggle did not allocate a CUDA GPU')
    (base/'hardware.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu_count':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(),
        'vram_bytes':torch.cuda.get_device_properties(0).total_memory},indent=2))
    data=base/'official_data'
    if len(archives)==1 and not data.exists():
        data.mkdir()
        with zipfile.ZipFile(archives[0]) as archive:
            for entry in archive.infolist():
                target=(data/entry.filename).resolve()
                if not target.is_relative_to(data):raise ValueError('Unsafe archive path')
            archive.extractall(data)
    elif len(mounted_train)==1:
        # Some Kaggle dataset ingestion paths expand uploaded ZIPs automatically.
        data=mounted_train[0].parents[3]
    reports=base/'reports';reports.mkdir(exist_ok=True)
    train=data/'annotations/task1/train/gt.csv';val=data/'annotations/task1/val/gt.csv'
    from upar.inspect_data import audit
    audit(train,val,data,reports/'data',workers=8)
    run('scripts/review_known_duplicates.py','--audit-dir',reports/'data')
    evidence=json.loads((reports/'data/data_audit.json').read_text())
    if not evidence['passed'] or evidence['leakage_review_required']:raise RuntimeError('Data gate failed')
    manifest=json.loads((reports/'data/image_manifest.json').read_text())
    images=[];ratios=set()
    for domain,number in [('Market1501',2),('PA100k',2),('PETA',1)]:
        count=0
        for name,item in manifest.items():
            ratio=round(item['size'][0]/item['size'][1],2)
            if name.startswith(domain+'/') and ratio not in ratios:
                ratios.add(ratio);images.append(data/name);count+=1
                if count==number:break
    os.environ['UPAR_TEST_IMAGE']=str(images[0])
    run('-m','pytest','-q','tests','--junitxml='+str(reports/'tests.xml'))
    from upar.parity import check
    parity=reports/'PRETRAINED_PARITY.json'
    check(images,'google/siglip2-base-patch16-naflex',parity,device='cuda')
    # Inference API/latency preflight uses actual pretrained weights with an
    # untrained PAR head. It is a timing/checkpoint test, never a competition model.
    import shutil
    stage=base/'inference_preflight';(stage/'assets').mkdir(parents=True,exist_ok=True)
    shutil.copy2(code/'submission_template/run.py',stage/'run.py')
    shutil.copytree(code/'upar',stage/'upar',dirs_exist_ok=True)
    shutil.copy2(reports/'foundation_native.pt',stage/'assets/best.pt')
    from upar.attributes import ATTRIBUTES
    (stage/'assets/calibration.json').write_text(json.dumps({'attributes':ATTRIBUTES,'thresholds':[.5]*40}))
    run('-m','upar.benchmark','--submission-dir',stage,'--images-dir',data/'PA100k',
        '--n',1000,'--batch-size',32,'--output',reports/'GPU_INFERENCE_PREFLIGHT.json')
    shutil.rmtree(stage)
    from upar.feasibility import benchmark
    feasibility=reports/'GPU_FEASIBILITY.json'
    batch=16
    try:
        speed=benchmark(str(train),str(data),str(parity),str(feasibility),batch_size=batch)
    except torch.cuda.OutOfMemoryError:
        # Same model and recipe; reduce microbatch only, preserve effective batch 64.
        torch.cuda.empty_cache();batch=8
        speed=benchmark(str(train),str(data),str(parity),str(feasibility),batch_size=batch)
    planned_train_hours=speed['epoch_train_seconds_estimate']*12/3600
    if planned_train_hours>8.0:
        raise RuntimeError(f'Training alone estimates {planned_train_hours:.2f}h; inspect benchmark before spending full session')
    # Foundation cache reuses the same revision. Only official train split is optimized.
    resumes=list(Path('/kaggle/input').rglob('last.pt'))
    if len(resumes)>1:raise RuntimeError('Multiple resume checkpoints attached; choose one source')
    resume_args=['--resume',str(resumes[0])] if resumes else []
    if resumes:
        import shutil
        destination=base/'runs/siglip2';destination.mkdir(parents=True,exist_ok=True)
        # Preserve previous best independently of the new session's latest state.
        for name in ['best.pt','best_val_predictions.npz','best_val_metrics.json','attribute_metrics.json','history.jsonl']:
            previous=resumes[0].parent/name
            if previous.exists():shutil.copy2(previous,destination/name)
    run('-m','upar.train','--train-csv',train,'--val-csv',val,'--data-root',data,
        '--output',base/'runs/siglip2','--batch-size',batch,'--accumulate',64//batch,
        '--epochs',12,'--workers',4,'--data-report',reports/'data/data_audit.json',
        '--parity-report',parity,'--feasibility-report',feasibility,'--max-hours',9.0,*resume_args)
    run('-m','upar.calibrate','--predictions',base/'runs/siglip2/best_val_predictions.npz',
        '--output',base/'runs/siglip2/calibration.json')
    # The file is a candidate until offline benchmark and official scoring checks are complete.
    candidate=base/'submission_candidate.zip'
    run('-m','upar.package','--checkpoint',base/'runs/siglip2/best.pt',
        '--calibration',base/'runs/siglip2/calibration.json','--output',candidate)
    submission=base/'submission_candidate';submission.mkdir(exist_ok=True)
    with zipfile.ZipFile(candidate) as archive:archive.extractall(submission)
    run('-m','upar.benchmark','--submission-dir',submission,'--images-dir',data/'PA100k',
        '--n',2000,'--batch-size',32,'--output',reports/'TRAINED_INFERENCE_BENCHMARK.json')
    (base/'STATUS.json').write_text(json.dumps({'training_completed':True,
        'candidate_created':True,'submitted':False,
        'limitations':['Official scorer implementation unavailable; formulas only verified.',
                      'Kaggle GPU benchmark does not certify official A10 300-second runtime.']}))
    # Keep weights, reports and config, release large cache/extracted data from saved outputs.
    # Never delete source datasets. These are derived copies under /kaggle/working.
    import shutil
    if data.resolve().is_relative_to(base):shutil.rmtree(data)
    shutil.rmtree(base/'hf-cache',ignore_errors=True)


if __name__=='__main__':main()
