"""PIL/NumPy NaFlex patch processing (no transformers or torchvision needed).

The layout and scaling replicate Siglip2ImageProcessor (patch_size=16).
Parity with the installed Hugging Face processor must be checked in prepare.py.
"""
from __future__ import annotations
import math
import random
import io
from functools import lru_cache
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


@lru_cache(maxsize=256)
def target_size(height, width, patch_size=16, max_num_patches=256):
    if min(height, width, patch_size, max_num_patches) <= 0:
        raise ValueError('Image dimensions and patch budget must be positive')
    def scaled(sz, factor):
        return max(patch_size, math.ceil(sz * factor / patch_size) * patch_size)
    low, high = 1e-5/10, 100.0
    while high - low >= 1e-5:
        mid = (high + low) / 2
        h, w = scaled(height, mid), scaled(width, mid)
        if (h // patch_size) * (w // patch_size) <= max_num_patches:
            low = mid
        else:
            high = mid
    return scaled(height, low), scaled(width, low)


def train_augment(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    # Mild geometric jitter without crop-out of most of the person.
    if random.random() < 0.65:
        w, h = img.size
        px, py = max(1, round(.04*w)), max(1, round(.04*h))
        img = ImageOps.expand(img, border=(px,py,px,py), fill=(0,0,0))
        left = random.randrange(0, 2*px+1)
        top = random.randrange(0, 2*py+1)
        img = img.crop((left,top,left+w,top+h))
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.88, 1.12))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.88,1.12))
    if random.random() < 0.10:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(.2,.6)))
    if random.random() < 0.15:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=random.randint(75, 95))
        buf.seek(0)
        with Image.open(buf) as degraded:
            img = degraded.convert('RGB')
    if random.random() < 0.05:
        # At most 0.5% of the image; avoid removing a whole accessory/person part.
        w, h = img.size
        ew, eh = max(1, round(w*.05)), max(1, round(h*.10))
        x, y = random.randint(0,w-ew), random.randint(0,h-eh)
        img.paste((127,127,127), (x,y,x+ew,y+eh))
    return img


def preprocess_image(img, max_patches=256, patch_size=16, augment=False):
    img = img.convert("RGB")
    if augment:
        img = train_augment(img)
    w, h = img.size
    h2, w2 = target_size(h, w, patch_size=patch_size, max_num_patches=max_patches)
    img = img.resize((w2, h2), Image.Resampling.BILINEAR)
    # SigLIP2 normalizes pixels with mean=0.5, std=0.5.
    array = np.asarray(img).astype(np.float64) * (1.0 / 255.0)
    array = (array.astype(np.float32) - np.float32(.5)) / np.float32(.5)
    ph, pw = h2//patch_size, w2//patch_size
    # HF Siglip2ImageProcessor: grid H, grid W, patch H, patch W, RGB.
    # Source: transformers/v4.57.1/models/siglip2/image_processing_siglip2.py
    patches = array.reshape(ph,patch_size,pw,patch_size,3).transpose(0,2,1,3,4)
    patches = np.ascontiguousarray(patches.reshape(ph*pw,3*patch_size*patch_size))
    valid = np.zeros(max_patches, dtype=np.bool_)
    valid[:ph*pw] = True
    padded = np.zeros((max_patches,3*patch_size*patch_size),dtype=np.float32)
    padded[:ph*pw] = patches
    return torch.from_numpy(padded), torch.from_numpy(valid), torch.tensor([ph,pw],dtype=torch.long)
