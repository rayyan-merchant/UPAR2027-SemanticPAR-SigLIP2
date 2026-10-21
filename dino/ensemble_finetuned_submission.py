"""Candidate B fine-tuned offline API. Copy as run.py beside assets/upar/dino/licenses.

Uses RAW branch probabilities, not the original production-calibrated SigLIP
probabilities. One separately preserved ensemble threshold vector follows mixing.
"""
import json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from upar.model import UPARModel,VisionConfig
from upar.attributes import ATTRIBUTES
from upar.preprocess import preprocess_image
from dino.v3_api import DinoV3Backbone,AttributeHead,preprocess,pool_descriptors

HERE=Path(__file__).resolve().parent
BATCH_SIZE=int(os.environ.get('UPAR_BATCH_SIZE','64'))
_SIGLIP=None;_DINO=None;_HEAD=None;_LOGIT_THRESHOLD=None;_DEVICE=None;_POOL=None

def blend_and_calibrate(siglip_logits,dino_logits,logit_threshold):
    mixture=.75*siglip_logits.float().sigmoid()+.25*dino_logits.float().sigmoid()
    # Monotone threshold shift maps source-calibrated decisions to official0.5.
    result=torch.sigmoid(torch.logit(mixture.clamp(1e-7,1-1e-7))-logit_threshold)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite ensemble probabilities')
    return result.clamp(0,1)

def load_model():
    global _SIGLIP,_DINO,_HEAD,_LOGIT_THRESHOLD,_DEVICE,_POOL
    if _SIGLIP is not None:return
    _DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sig=torch.load(HERE/'assets/siglip_best.pt',map_location='cpu',weights_only=False)
    if sig['attributes']!=ATTRIBUTES or sig['max_patches']!=256:raise ValueError('SigLIP contract differs')
    model=UPARModel(VisionConfig(**sig['vision_config']));model.load_state_dict(sig['state_dict'],strict=True)
    dino=torch.load(HERE/'assets/dino.pt',map_location='cpu',weights_only=False)
    if dino['attributes']!=ATTRIBUTES or dino['canvas']!=[320,160]:raise ValueError('DINO deployment contract differs')
    backbone=DinoV3Backbone();backbone.load_state_dict(dino['backbone_state_dict'],strict=True)
    head=AttributeHead();head.load_state_dict(dino['head_state_dict'],strict=True)
    cal=json.loads((HERE/'assets/ensemble_calibration.json').read_text())
    if cal['attributes']!=ATTRIBUTES or cal['siglip_raw_probability_weight']!=.75:raise ValueError('Ensemble recipe differs')
    thresholds=np.asarray(cal['thresholds'],np.float32)
    if thresholds.shape!=(40,) or not ((thresholds>0)&(thresholds<1)).all():raise ValueError('Invalid source-only thresholds')
    _LOGIT_THRESHOLD=torch.tensor(np.log(thresholds/(1-thresholds)),device=_DEVICE)
    _DINO=backbone.eval().to(_DEVICE);_HEAD=head.eval().to(_DEVICE)
    _POOL=ThreadPoolExecutor(max_workers=min(8,os.cpu_count() or 1))
    _SIGLIP=model.eval().to(_DEVICE)

def prepare_sample(sample):
    if sample.get('attribute_names')!=ATTRIBUTES:raise ValueError('Unexpected attribute order')
    with Image.open(sample['image_path']) as source:
        image=source.convert('RGB');patches,mask,shape=preprocess_image(image,max_patches=256)
        pixels,box=preprocess(image)
    return patches,mask,shape,pixels,box

@torch.inference_mode()
def predict_batch(samples):
    if not samples:return []
    if BATCH_SIZE<=0:raise ValueError('Batch size must be positive')
    if _SIGLIP is None:load_model()
    result=[]
    for start in range(0,len(samples),BATCH_SIZE):
        prepared=list(_POOL.map(prepare_sample,samples[start:start+BATCH_SIZE]))
        patches,mask,shape,pixels,boxes=zip(*prepared)
        patches=torch.stack(patches).to(_DEVICE);mask=torch.stack(mask).to(_DEVICE);shape=torch.stack(shape)
        pixels=torch.stack(pixels).to(_DEVICE);boxes=torch.stack(boxes).to(_DEVICE)
        with torch.autocast('cuda',dtype=torch.float16,enabled=_DEVICE.type=='cuda'):
            s=_SIGLIP(patches,mask,shape)
            # Fine-tuning used FP32 pooled descriptors; no frozen-cache round-trip.
            d=_HEAD(pool_descriptors(_DINO(pixels),boxes))
        p=blend_and_calibrate(s,d,_LOGIT_THRESHOLD)
        result.extend(p.cpu().tolist())
    if len(result)!=len(samples) or any(len(row)!=40 for row in result):raise AssertionError('Incorrect output shape')
    return result

def predict_image(sample):return predict_batch([sample])[0]
