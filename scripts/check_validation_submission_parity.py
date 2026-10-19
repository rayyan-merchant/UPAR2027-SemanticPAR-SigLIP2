"""Compare actual validation evaluator with immutable accepted ZIP; CPU only."""
import hashlib,importlib,importlib.util,json,sys,types,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'diagnosis_1003877';torch.set_num_threads(4)
# Separate namespace ensures the training-path implementation is independent
# of the package imported by the submitted run.py.
package=types.ModuleType('diagnosis_source');package.__path__=[str(ROOT/'upar')];sys.modules['diagnosis_source']=package
source_model=importlib.import_module('diagnosis_source.model');source_data=importlib.import_module('diagnosis_source.data');source_train=importlib.import_module('diagnosis_source.train')
pre=importlib.import_module('diagnosis_source.preprocess')
with zipfile.ZipFile(ROOT.parent/'submission/UPAR2027_SemanticPAR_SigLIP2.zip') as archive:
    # Reuse the fully verified extraction; never duplicate large weights.
    unpacked=ROOT.parent/'verification-torch241-py311/submission'
    for name in archive.namelist():
        if name.endswith('/'):
            continue
        digest=hashlib.sha256()
        with archive.open(name) as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        local=hashlib.sha256()
        with (unpacked/name).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                local.update(block)
        assert local.digest()==digest.digest(),name
    source_equal={name:archive.read('upar/'+name)==(ROOT/'upar'/name).read_bytes() for name in ['model.py','preprocess.py','attributes.py']}
assert all(source_equal.values())
sys.path.insert(0,str(unpacked));spec=importlib.util.spec_from_file_location('accepted_run',unpacked/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
run.load_model();assert run._DEVICE.type=='cpu'
ck=torch.load(unpacked/'assets/best.pt',map_location='cpu',weights_only=False)
model=source_model.UPARModel(source_model.VisionConfig(**ck['vision_config']));model.load_state_dict(ck['state_dict']);model.eval()
z=np.load(ROOT.parent/'kaggle-v4-artifacts/runs/siglip2/best_val_predictions.npz');rows=source_data.load_rows('D:/UPAR2027/official/data/annotations/task1/val/gt.csv')
assert [r[0] for r in rows]==z['image_paths'].tolist()
assert np.array_equal(np.stack([r[1] for r in rows]),z['y'])
assert np.array_equal(np.stack([r[2] for r in rows]),z['valid'])
assert [r[3] for r in rows]==z['domains'].tolist()
manifest=json.loads((ROOT/'reports/data_mapped/image_manifest.json').read_text())
indices=[]
for domain in sorted(set(z['domains'])):
    candidates=np.flatnonzero(z['domains']==domain)
    # Deterministic spread includes varied source paths/aspect ratios.
    indices.extend(candidates[np.linspace(0,len(candidates)-1,16,dtype=int)].tolist())
actual=[];selected=[];padding=[];pixels_equal=True
for i in indices:
    name=rows[i][0];path=Path('D:/UPAR2027/official/data')/manifest[name]['source_relative_path'];actual.append(path)
    selected.append((manifest[name]['source_relative_path'],rows[i][1],rows[i][2],rows[i][3]))
    with Image.open(path) as image:
        a=pre.preprocess_image(image,max_patches=256);b=run.preprocess_image(image,max_patches=256)
    pixels_equal=pixels_equal and all(torch.equal(x,y) for x,y in zip(a,b));padding.append(256-int(a[1].sum()))
ds=source_data.UPARDataset('D:/UPAR2027/official/data/annotations/task1/val/gt.csv','D:/UPAR2027/official/data',256,augment=False);ds.rows=selected
loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
raw,labels,valid,domains=source_train.predict(model,loader,torch.device('cpu'),amp=False)
samples=[{'image_path':str(p),'attribute_names':run.ATTRIBUTES} for p in actual]
api=np.asarray(run.predict_batch(samples));t=np.asarray(json.loads((unpacked/'assets/calibration.json').read_text())['thresholds'],np.float32)
# Recover raw logits through the identical accepted backbone/head with no offsets.
with torch.inference_mode():
    api_raw=[];expected=[]
    for start in range(0,len(samples),32):
        pieces=[run.preprocess_image(Image.open(p).convert('RGB'),max_patches=256) for p in actual[start:start+32]]
        p,m,s=[torch.stack([piece[k] for piece in pieces]) for k in range(3)]
        logits=run._MODEL(p,m,s).float();api_raw.extend(torch.sigmoid(logits).numpy());expected.extend(torch.sigmoid(logits-run._THRESHOLDS).numpy())
    # Alter only invalid padding in three actual-image tensors.
    pieces=[run.preprocess_image(Image.open(p).convert('RGB'),max_patches=256) for p in actual[:3]]
    p,m,s=[torch.stack([piece[k] for piece in pieces]) for k in range(3)]
    baseline=run._MODEL(p,m,s);changed=p.clone();changed[~m]=torch.randn_like(changed[~m])*10
    altered=run._MODEL(changed,m,s);padding_error=float((baseline-altered).abs().max())
api_raw=np.asarray(api_raw);expected=np.asarray(expected)
error=float(np.max(np.abs(raw-api_raw)));offset_error=float(np.max(np.abs(api-expected)))
stored=z['prob'][indices];amp_error=np.abs(raw-stored)
result={'passed':pixels_equal and error<1e-6 and offset_error<1e-6 and padding_error<1e-6,
        'actual_images':len(indices),'domains':sorted(set(domains)),'python':sys.version,'torch':torch.__version__,
        'source_code_matches_archive':source_equal,'full_validation_paths_labels_valid_domains_equal':True,
        'preprocessing_exact':pixels_equal,'padded_patch_counts':padding,'native_validation_vs_zip_raw_max_abs':error,
        'threshold_offset_max_abs':offset_error,'offset_decision_mismatches':int(((raw>t)!=(api>.5)).sum()),
        'invalid_padding_perturbation_max_logit_error':padding_error,
        'stored_GPU_AMP_vs_CPU_FP32_probabilities':{'max_abs':float(amp_error.max()),'mean_abs':float(amp_error.mean()),
          'raw_decision_disagreements':int(((raw>.5)!=(stored>.5)).sum()),'calibrated_decision_disagreements':int(((raw>t)!=(stored>t)).sum()),'cells':int(raw.size)},
        'limitation':'48 actual known validation images; stored logits came from torch2.11 T4 AMP, this comparison is torch2.4.1 CPU FP32. Not a full validation rerun or hidden data inspection.'}
(OUT/'validation_submission_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
assert result['passed']
