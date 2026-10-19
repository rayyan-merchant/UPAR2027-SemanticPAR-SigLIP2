"""UPAR2027 Track1 official code submission API; torch/Pillow/numpy only."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from upar.model import UPARModel, VisionConfig
from upar.preprocess import preprocess_image
from upar.attributes import ATTRIBUTES

HERE = Path(__file__).resolve().parent
BATCH_SIZE = int(os.environ.get('UPAR_BATCH_SIZE', '32'))
_MODEL=None
_THRESHOLDS=None
_DEVICE=None
_PATCHES=256


def load_model() -> None:
    global _MODEL,_THRESHOLDS,_DEVICE,_PATCHES
    if _MODEL is not None: return
    _DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(HERE/'assets'/'best.pt',map_location='cpu',weights_only=False)
    if ckpt['attributes'] != ATTRIBUTES:
        raise ValueError('Checkpoint attributes differ from official order')
    _PATCHES=int(ckpt['max_patches'])
    model=UPARModel(VisionConfig(**ckpt['vision_config']))
    model.load_state_dict(ckpt['state_dict'],strict=True)
    model.eval().to(_DEVICE)
    cal=HERE/'assets'/'calibration.json'
    if cal.exists():
        calibration=json.loads(cal.read_text())
        if calibration.get('attributes') != ATTRIBUTES:
            raise ValueError('Calibration attributes differ from official order')
        thresholds=np.asarray(calibration['thresholds'],dtype=np.float32)
    else:
        thresholds=np.full(40,.5,np.float32)
    if thresholds.shape!=(40,) or not (0<thresholds).all() or not (thresholds<1).all():
        raise ValueError('Invalid calibration thresholds')
    _THRESHOLDS=torch.tensor(np.log(thresholds/(1-thresholds)),device=_DEVICE,dtype=torch.float32)
    _MODEL=model


@torch.inference_mode()
def predict_batch(samples: list[dict]) -> list[list[float]]:
    if not samples:return []
    if BATCH_SIZE<=0:raise ValueError('UPAR_BATCH_SIZE must be positive')
    results=[]
    for start in range(0,len(samples),BATCH_SIZE):
        results.extend(_predict_chunk(samples[start:start+BATCH_SIZE]))
    return results


@torch.inference_mode()
def _predict_chunk(samples):
    if _MODEL is None:load_model()
    patches=[];masks=[];shapes=[]
    for s in samples:
        if s.get('attribute_names') != ATTRIBUTES:
            raise ValueError('Unexpected ordered attribute names')
        with Image.open(s['image_path']) as im:
            px,mask,shape=preprocess_image(im,max_patches=_PATCHES)
        patches.append(px);masks.append(mask);shapes.append(shape)
    px=torch.stack(patches).to(_DEVICE,non_blocking=True)
    mask=torch.stack(masks).to(_DEVICE,non_blocking=True)
    # Shapes stay on CPU: position interpolation uses Python sizes; avoid per-image GPU sync.
    shape=torch.stack(shapes)
    with torch.autocast('cuda',dtype=torch.float16,enabled=_DEVICE.type=='cuda'):
        logits=_MODEL(px,mask,shape).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite model output')
    probs=torch.sigmoid(logits-_THRESHOLDS).clamp(0,1)
    results=probs.cpu().tolist()
    if len(results)!=len(samples) or any(len(r)!=40 for r in results):
        raise AssertionError('Incorrect output shape')
    return results


def predict_image(sample: dict) -> list[float]:
    return predict_batch([sample])[0]
