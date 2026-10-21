"""Full cold-process offline inference benchmark, no score/label access.

Run with exact package folder, permitted image manifest and output path. Image
staging is outside the official ingestion workload; import/load/decode/forward/
mix/calibration/JSON serialization are inside the timed child process.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def main():
    a=argparse.ArgumentParser();a.add_argument('--package',required=True);a.add_argument('--images',required=True);a.add_argument('--output',required=True);a.add_argument('--child',action='store_true');p=a.parse_args()
    if not p.child:
        begin=time.perf_counter()
        proc=subprocess.run([sys.executable,__file__,'--child','--package',p.package,'--images',p.images,'--output',p.output],timeout=300,check=True)
        report=json.loads(Path(p.output).read_text());report['cold_python_process_wall_seconds']=time.perf_counter()-begin
        report['passes_240s_complete_workload']=report['cold_python_process_wall_seconds']<=240 and report['images']==29248
        report['official_300s_limit_pass']=report['cold_python_process_wall_seconds']<300
        report['deployment_runtime_gate_passed']=report['passes_240s_complete_workload'] and report['A10_equivalent_environment_confirmed']
        Path(p.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
    begin=time.perf_counter()
    import importlib.abc,socket
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,*args):
            if fullname.split('.')[0] in {'transformers','huggingface_hub','timm','safetensors','requests'}:raise ImportError('Non-offline model dependency '+fullname)
    sys.meta_path.insert(0,Block())
    import torch,numpy as np
    def deny(*a,**kw):raise RuntimeError('Network forbidden during benchmark')
    socket.socket=deny
    sys.path.insert(0,str(Path(p.package).resolve()));import run
    from upar.attributes import ATTRIBUTES
    image_paths=json.loads(Path(p.images).read_text())
    if len(image_paths)!=29248 or len(set(image_paths))!=29248:raise ValueError('Require29248 distinct permitted representative images')
    if not torch.cuda.is_available():raise RuntimeError('CPU timing cannot certify GPU deployment')
    run.load_model();torch.cuda.synchronize();loaded=time.perf_counter();serialize_seconds=0.;h=hashlib.sha256();rows=0;output_bytes=0
    with torch.inference_mode():
        for start in range(0,len(image_paths),run.BATCH_SIZE):
            samples=[{'attribute_names':ATTRIBUTES,'image_path':path} for path in image_paths[start:start+run.BATCH_SIZE]]
            probs=run.predict_batch(samples)
            if len(probs)!=len(samples) or not np.isfinite(probs).all():raise ValueError('Invalid prediction batch')
            tick=time.perf_counter();blob=(json.dumps(probs,separators=(',',':'))+'\n').encode();serialize_seconds+=time.perf_counter()-tick;h.update(blob);output_bytes+=len(blob);rows+=len(samples)
    torch.cuda.synchronize();finished=time.perf_counter();gpu=torch.cuda.get_device_name();version_ok=torch.__version__.split('+')[0]=='2.4.1' and sys.version_info[:2]==(3,11)
    report={'images':rows,'distinct_images':len(set(image_paths)),'complete_inner_seconds':finished-begin,'import_and_model_loading_seconds':loaded-begin,'decoding_preprocessing_forward_mix_calibration_serialization_seconds':finished-loaded,'serialization_seconds':serialize_seconds,'serialized_bytes':output_bytes,'prediction_stream_sha256':h.hexdigest(),'gpu':gpu,'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'cpu_count':os.cpu_count(),'batch_size':run.BATCH_SIZE,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'network_disabled':True,'labels_accessed':False,'A10_equivalent_environment_confirmed':('A10' in gpu and version_ok),'surrogate_only_if_not_A10':True,'official_upload_performed':False}
    Path(p.output).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
