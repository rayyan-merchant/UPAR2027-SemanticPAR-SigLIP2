"""Candidate C's single predeclared augmentation-only intervention.

Original inference preprocessing and production training defaults stay intact.
Install explicitly in isolated candidate C training/probe processes only.
"""
import io,random
import numpy as np
from PIL import Image,ImageEnhance,ImageOps,ImageFilter

POLICY={'name':'surveillance_v1','mirror':{'p':.5},'original_pad_crop':{'p':.65,'fraction':.04},'brightness':{'p':.5,'range':[.90,1.10]},'contrast':{'p':.5,'range':[.88,1.12]},'down_up':{'p':.30,'scale':[.65,1.]},'blur':{'p':.15,'sigma':[.2,.7]},'jpeg':{'p':.30,'quality':[60,95]},'gamma':{'p':.20,'range':[.90,1.10]},'original_small_erase':{'p':.05,'width_fraction':.05,'height_fraction':.10}}

def train_augment(img):
    img=img.convert('RGB')
    if random.random()<.5:img=ImageOps.mirror(img)
    if random.random()<.65:
        w,h=img.size;px,py=max(1,round(.04*w)),max(1,round(.04*h))
        img=ImageOps.expand(img,border=(px,py,px,py),fill=(0,0,0))
        x,y=random.randrange(0,2*px+1),random.randrange(0,2*py+1);img=img.crop((x,y,x+w,y+h))
    if random.random()<.5:img=ImageEnhance.Brightness(img).enhance(random.uniform(.90,1.10))
    if random.random()<.5:img=ImageEnhance.Contrast(img).enhance(random.uniform(.88,1.12))
    if random.random()<.30:
        w,h=img.size;scale=random.uniform(.65,1.)
        img=img.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BICUBIC).resize((w,h),Image.Resampling.BICUBIC)
    if random.random()<.15:img=img.filter(ImageFilter.GaussianBlur(random.uniform(.2,.7)))
    if random.random()<.30:
        b=io.BytesIO();img.save(b,format='JPEG',quality=random.randint(60,95));b.seek(0)
        with Image.open(b) as decoded:img=decoded.convert('RGB')
    if random.random()<.20:
        gamma=random.uniform(.90,1.10);lut=np.clip(np.round(255*(np.arange(256)/255.)**gamma),0,255).astype(np.uint8).tolist()
        img=img.point(lut*3)
    if random.random()<.05:
        w,h=img.size;ew,eh=max(1,round(w*.05)),max(1,round(h*.10));x,y=random.randint(0,w-ew),random.randint(0,h-eh)
        img.paste((127,127,127),(x,y,x+ew,y+eh))
    return img

def install():
    from upar import preprocess
    preprocess.train_augment=train_augment
