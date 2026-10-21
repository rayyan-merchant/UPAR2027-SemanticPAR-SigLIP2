from pathlib import Path
import argparse
import base64,io,json,zipfile,hashlib,ast
r=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
b=io.BytesIO();hashes={}
with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['dino','upar','scripts','kaggle','tests']:
  for p in sorted((r/folder).glob('*.py')):
   ast.parse(p.read_text(encoding='utf-8-sig'));n=p.relative_to(r).as_posix();z.write(p,n);hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
 for name in ['pytest.ini','reports/lodo/THREEFOLD_ARCHIVES.json','reports/dino_v3/PLAN.json','reports/dino_v3/GATE_BUILD_RECEIPT.json','reports/dino_v3/GATE_FEASIBILITY.json','reports/dino_v3/GATE_PARITY.json','reports/dino_v3/GATE_HARDWARE.json']:
  z.write(r/name,name);hashes[name]=hashlib.sha256((r/name).read_bytes()).hexdigest()
code="""import base64,io,zipfile,sys,subprocess,os
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
code=Path('/kaggle/working/dinov3_code');code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(code)
child=code/'frozen_child.py'
child.write_text(\"import subprocess,sys,json,importlib.metadata as md\\nfrom pathlib import Path\\npackages=['torch','transformers','tokenizers','huggingface_hub','Pillow','numpy']\\nPath('/kaggle/working/ENVIRONMENT_BEFORE.json').write_text(json.dumps({k:md.version(k) for k in packages},indent=2))\\nsubprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','tokenizers>=0.22,<0.23','huggingface_hub>=0.34,<1','safetensors','pytest'],check=True,timeout=150)\\nPath('/kaggle/working/ENVIRONMENT_AFTER.json').write_text(json.dumps({k:md.version(k) for k in packages},indent=2))\\nassert md.version('transformers')=='4.57.1'\\nsubprocess.run([sys.executable,'-m','pytest','tests/test_dino_v3.py','tests/test_dino.py','tests/test_correctness.py','-q','-p','no:cacheprovider','--junitxml=/kaggle/working/NATIVE_TESTS.xml'],check=True,timeout=120)\\nsubprocess.run([sys.executable,'-m','dino.v3_runner'],check=True)\\n\")
subprocess.run([sys.executable,str(code/'kaggle/pretraining_watchdog.py'),'--marker','/kaggle/working/TRAINING_START.json','--report','/kaggle/working/PRETRAINING_WATCHDOG.json','--max-seconds','600','--',sys.executable,str(child)],cwd=code,check=True)
""".replace('PAYLOAD',repr(base64.b64encode(b.getvalue()).decode()))
(out/'run_frozen.py').write_text(code)
m={'id': 'rayyannmerchant/upar2027-dinov3-timm-feasibility', 'title': 'UPAR2027 DINOv3 timm feasibility', 'code_file': 'run_gate.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True, 'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f', 'docker_image_pinning_type': 'original', 'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': ['rayyannmerchant/upar2027-private-paheld-archives', 'rayyannmerchant/upar2027-private-lodo-supplement', 'rayyannmerchant/upar2027-private-dinov3-foundation'], 'kernel_sources': ['rayyannmerchant/upar2027-threefold-archives-resource-safe'], 'competition_sources': []};m.update(id='rayyannmerchant/upar2027-dinov3-frozen-threefold',title='UPAR2027 DINOv3 frozen threefold',code_file='run_frozen.py');m['kernel_sources'].insert(0,'rayyannmerchant/upar2027-siglip2-semanticpar');(out/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
receipt={'kernel':m['id'],'status':'PREPARED_NOT_LAUNCHED','script_sha256':hashlib.sha256(code.encode()).hexdigest(),'source_sha256':hashes,'gpu_gate_reported_before_extraction':True,'reference_gate':'rayyannmerchant/upar2027-dinov3-timm-feasibility/v1 COMPLETE','remaining_quota_displayed_hours':10.81,'cumulative_diagnostic_cap_hours':2,'previous_dino_used_displayed_approx_hours':.16,'new_provider_timeout_seconds':6500,'new_internal_budget_seconds':6200,'all_domain_training_allowed':False,'competition_upload_allowed':False}
(r/'reports/dino_v3/FROZEN_BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'kernel':m['id'],'script_sha256':receipt['script_sha256'],'code_bytes':len(code)}))
