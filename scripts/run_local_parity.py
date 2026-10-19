"""Select five audited real images with distinct aspect ratios and check FP32 parity."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from upar.parity import check

root=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(root.parents[1]/'work/hf-cache'))
torch.set_num_threads(4)
manifest=json.loads((root/'reports/data_mapped/image_manifest.json').read_text())
images=[];ratios=set()
for domain in ['Market1501','PA100k','PETA']:
    count=0
    for name,item in manifest.items():
        ratio=round(item['size'][0]/item['size'][1],2)
        if name.startswith(domain+'/') and ratio not in ratios:
            ratios.add(ratio);images.append('D:/UPAR2027/official/data/'+item['source_relative_path'])
            count+=1
            if count>=(2 if domain!='PETA' else 1):break
check(images,'google/siglip2-base-patch16-naflex',root/'reports/PRETRAINED_PARITY.json',device='cpu')
