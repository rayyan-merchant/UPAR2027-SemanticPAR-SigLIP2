"""Bounded DINOv3 benchmark ONLY: never extracts full features or trains heads."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from safetensors.torch import load_file
from dino.v3_model import DinoV3Backbone
from dino.model import preprocess,pool_descriptors
BASE=Path('/kaggle/working');OUT=BASE/'dinov3_gate';CODE=Path(__file__).resolve().parents[1]
SHA='1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def find(name):
    found=[]
    for folder,dirs,files in os.walk('/kaggle/input'):
        dirs[:]=[d for d in dirs if d not in {'staged_data','archive_cache','official_data','hf-cache','runs','submission_candidate','upar_code'}]
        if name in files:found.append(Path(folder)/name)
    return found

class Images(Dataset):
    def __init__(self,paths):self.paths=paths
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(OUT/'staged_data'/self.paths[i]) as im:x,box=preprocess(im,canvas=(320,160))
        return x,box

def backbone():
    m=DinoV3Backbone();m.load_state_dict(load_file(str(OUT/'weights/model.safetensors')),strict=True)
    return m.eval().requires_grad_(False).cuda()

def worker(tag):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1
    paths=json.loads((OUT/'SOURCE_SAMPLES.json').read_text())['paths'];m=backbone()
    loader=DataLoader(Images(paths*10),batch_size=64,num_workers=2,pin_memory=True,timeout=30)
    measured=0;start=None;data_wait=0.;end=time.perf_counter();last=None
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,(x,box) in enumerate(loader):
            fetched=time.perf_counter()
            if i==2:torch.cuda.synchronize();start=time.perf_counter();data_wait=0.
            with torch.autocast('cuda',dtype=torch.float16):t=m(x.cuda(non_blocking=True))
            z=pool_descriptors(t,box.cuda(),grid=(20,10),patch=16)
            if not torch.isfinite(z).all():raise ValueError('Nonfinite descriptors')
            if i>=2:measured+=len(x);data_wait+=max(0,fetched-end)
            end=time.perf_counter()
            if i>=33:break
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    write(OUT/(tag+'.json'),{'images':measured,'seconds':seconds,'images_per_second':measured/seconds,'data_wait_seconds':data_wait,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'batch':64,'workers':2,'threads':2,'device':torch.cuda.get_device_name(0),'mode':'decode+aspect_preprocess+native_AMP_forward+6descriptor_pool','warmed_batches':2,'unique_source_samples':len(paths),'full_extraction':False,'training':False})

def main():
    import timm,psutil
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Require2T4')
    if shutil.disk_usage(BASE).free<8*1024**3:raise RuntimeError('Insufficient working disk')
    write(OUT/'HARDWARE.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'timm':timm.__version__,'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'cpu_count':os.cpu_count(),'free_disk':shutil.disk_usage(BASE).free})
    weights=find('RECEIPT.json');weights=[p for p in weights if json.loads(p.read_text()).get('repo')=='timm/vit_base_patch16_dinov3.lvd1689m']
    if len(weights)!=1:raise ValueError('Need one pinned DINOv3 dataset')
    dest=OUT/'weights';dest.mkdir(exist_ok=True)
    for n in ['model.safetensors','config.json','DINOV3_LICENSE.txt','TIMM_APACHE_2_0.txt']:shutil.copyfile(weights[0].parent/n,dest/n)
    assert digest(dest/'model.safetensors')==SHA
    coverage=json.loads((CODE/'reports/lodo/THREEFOLD_ARCHIVES.json').read_text());chosen=[]
    for item in coverage['archives']:
        candidates=find(item['archive']);assert len(candidates)==1
        chosen.append(dict(item,path=str(candidates[0])))
    sys.path.insert(0,str(CODE/'kaggle'));from archive_stage import stage
    staging=stage(chosen,OUT/'staged_data',OUT/'STAGING.json',120,30,120,0)
    assert staging['images']==131076 and staging['path_set_sha256']==coverage['coverage']['required_path_set_sha256']
    sources=[p for p in find('source_train.csv') if p.parent.parent.name=='PA100k'];assert len(sources)==1
    expected=json.loads((CODE/'reports/dino_v3/PLAN.json').read_text())['expected_splits']['PA100k']['partitions']['source_train']['sha256']
    assert digest(sources[0])==expected
    bydomain={}
    with sources[0].open() as f:
        rows=csv.reader(f);next(rows)
        for row in rows:bydomain.setdefault(row[0].split('/')[0],[]).append(row[0])
    assert set(bydomain)=={'Market1501','PETA'}
    samples=[bydomain[d][i] for i in range(128) for d in sorted(bydomain)]
    write(OUT/'SOURCE_SAMPLES.json',{'paths':samples,'source_only':True,'source_split_sha256':expected})
    native=backbone()
    # Same timm architecture and exact hash-verified safetensors used by the CPU
    # HF-entry-point reference; file override avoids a GPU-time network download.
    ref=timm.create_model('vit_base_patch16_dinov3',pretrained=True,num_classes=0,global_pool='avg',pretrained_cfg_overlay={'source':'','file':str(dest/'model.safetensors')}).eval().cuda()
    checks=[];x,box=next(iter(DataLoader(Images(samples[:4]),batch_size=4,num_workers=0)));x=x.cuda()
    for amp in [False,True]:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):a=native(x);b=ref.forward_features(x)
        da=pool_descriptors(a,box.cuda(),grid=(20,10),patch=16);db=pool_descriptors(b,box.cuda(),grid=(20,10),patch=16)
        error=float((a-b).abs().max());rel=float((a-b).norm()/b.norm());de=float((da-db).abs().max())
        checks.append({'amp':amp,'max_abs':error,'relative_l2':rel,'descriptor_max_abs':de,'tokens':list(a.shape),'descriptors':list(da.shape)})
        if error>(.05 if amp else 5e-4) or rel>(.005 if amp else 1e-5) or de>(.05 if amp else 5e-4):raise RuntimeError('Predeclared GPU parity tolerance failed')
    del native,ref,a,b,da,db,x;import gc;gc.collect();torch.cuda.empty_cache()
    (OUT/'PARITY.json').write_text(json.dumps({'passed':True,'comparisons':checks,'weights_sha256':SHA,'FP32_periods':True},indent=2))
    begin=time.monotonic();children=[];util=[]
    for gpu in [0,1]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        log=(OUT/f'gpu{gpu}.log').open('w');p=subprocess.Popen([sys.executable,'-m','dino.v3_feasibility','--worker',f'GPU{gpu}'],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append((p,log))
    try:
        while any(p.poll() is None for p,_ in children):
            if time.monotonic()-begin>150:raise TimeoutError('Short GPU benchmark exceeded150s')
            if any(p.poll() not in [None,0] for p,_ in children):raise RuntimeError('GPU benchmark worker failed')
            util.append({'elapsed':time.monotonic()-begin,'cpu_percent':psutil.cpu_percent(interval=1),'gpu_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used','--format=csv,noheader,nounits'],text=True).strip()})
    finally:
        for p,log in children:
            if p.poll() is None:os.killpg(p.pid,15)
            log.close()
    rates=[json.loads((OUT/f'GPU{i}.json').read_text()) for i in [0,1]]
    conservative_ips=min(x['images_per_second'] for x in rates)
    passed=conservative_ips>=75 and max(x['peak_GiB'] for x in rates)<=12 and 131076/conservative_ips<=1800
    write(OUT/'FEASIBILITY.json',{'passed':passed,'parity_passed':True,'workers':rates,'utilization_samples':util,'full_cache_bytes':131076*6*768*2,'single_worker_conservative_full_extract_seconds':131076/conservative_ips,'measured_sum_rate_full_extract_projection_seconds':131076/sum(x['images_per_second'] for x in rates),'two_workers_extraction_seconds_with_50percent_margin':1.5*131076/sum(x['images_per_second'] for x in rates),'measured_deployment_single_branch_29248_seconds_projection':29248/conservative_ips,'not_official_A10_deployment_proof':True,'full_extraction_started':False,'training_started':False})
    shutil.rmtree(OUT/'staged_data');shutil.rmtree(OUT/'archive_cache');shutil.rmtree(dest)
    write(BASE/'STATUS.json',{'phase':'benchmark_complete' if passed else 'benchmark_rejected','full_extraction_started':False,'training_started':False,'passed':passed})
    if not passed:raise RuntimeError('Throughput/memory gate failed')
    print('DINOv3 benchmark passed; deliberately stopping before full extraction or training.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker');args=ap.parse_args()
    if args.worker:worker(args.worker)
    else:main()
