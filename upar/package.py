"""Make fully offline CodaBench ZIP: run.py at archive root."""
from __future__ import annotations
import argparse
import shutil
import zipfile
import json
import torch
from .attributes import ATTRIBUTES
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--calibration',required=True)
    ap.add_argument('--output',default='submission.zip')
    args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if not ck.get('trained') or ck.get('attributes')!=ATTRIBUTES:
        raise ValueError('Competition packaging requires an actual trained checkpoint in official order')
    calibration=json.loads(Path(args.calibration).read_text())
    if calibration.get('attributes')!=ATTRIBUTES:
        raise ValueError('Calibration label ordering missing or incorrect')
    import numpy as np
    thresholds=np.asarray(calibration['thresholds'])
    if thresholds.shape!=(40,) or not np.isfinite(thresholds).all() or not ((thresholds>0)&(thresholds<1)).all():
        raise ValueError('Invalid calibration thresholds')
    root=Path(__file__).resolve().parent.parent
    template=root/'submission_template'
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        z.write(template/'run.py','run.py')
        z.write(template/'metadata.yaml','metadata.yaml')
        for p in (root/'upar').glob('*.py'):
            if p.name in ('__init__.py','model.py','preprocess.py','attributes.py'):
                z.write(p,'upar/'+p.name)
        z.write(args.checkpoint,'assets/best.pt',compress_type=zipfile.ZIP_STORED)
        z.write(args.calibration,'assets/calibration.json')
    with zipfile.ZipFile(output) as z:
        if 'run.py' not in z.namelist() or 'metadata.yaml' not in z.namelist():
            raise RuntimeError('Archive lacks required root files')
        print('Packaged:',output,'size MB:',round(output.stat().st_size/1048576,1))
        print('Contents:',[x for x in z.namelist() if not x.endswith('.pt')])
if __name__=='__main__': main()
