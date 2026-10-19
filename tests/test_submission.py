"""Temporary synthetic checkpoints test packaging/API; no competition score is implied."""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from upar.attributes import ATTRIBUTES
from upar.model import UPARModel,VisionConfig
from upar.metrics import macro_score

ROOT=Path(__file__).resolve().parents[1]


def test_offline_api_and_zip_on_actual_image(tmp_path):
    torch.set_num_threads(2)
    cfg=VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_patches=16)
    model=UPARModel(cfg).eval()
    checkpoint=tmp_path/'fixture.pt'
    # This tag exists only in an isolated temporary packaging test fixture.
    torch.save({'state_dict':model.state_dict(),'vision_config':cfg.as_dict(),'max_patches':16,
                'attributes':ATTRIBUTES,'trained':True,'test_fixture':True},checkpoint)
    cal=tmp_path/'calibration.json'
    cal.write_text(json.dumps({'attributes':ATTRIBUTES,'thresholds':[.5]*40}))
    archive=tmp_path/'fixture.zip'
    result=subprocess.run([sys.executable,'-m','upar.package','--checkpoint',str(checkpoint),
                           '--calibration',str(cal),'--output',str(archive)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    unpacked=tmp_path/'unpacked'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert {'run.py','metadata.yaml','assets/best.pt','assets/calibration.json'}<=set(names)
        assert not any('train.py' in n or 'transformers' in n for n in names)
        assert z.read('metadata.yaml')==(ROOT/'submission_template/metadata.yaml').read_bytes()
        z.extractall(unpacked)
    # Test is portable; on this audited host it uses a real annotated pedestrian image.
    real=Path(os.environ.get('UPAR_TEST_IMAGE','D:/UPAR2027/official/data/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg'))
    if not real.is_file():
        # Explicitly synthetic portable fixture, reported separately by preflight.
        real=tmp_path/'synthetic.png';Image.new('RGB',(32,64)).save(real)
    script=unpacked/'offline_test.py'
    script.write_text('''import importlib.abc,sys,socket,json,torch
torch.set_num_threads(2)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'transformers','huggingface_hub','timm','requests'}:
            raise ImportError('Blocked offline dependency: '+fullname)
sys.meta_path.insert(0,Block())
def denied(*args,**kwargs):raise RuntimeError('Network forbidden')
socket.socket=denied
import run
from upar.attributes import ATTRIBUTES
run.load_model()
sample={'image_path':sys.argv[1],'attribute_names':ATTRIBUTES}
assert run.predict_batch([])==[]
result=run.predict_batch([sample]*35)
assert len(result)==35 and all(len(row)==40 for row in result)
tolerance=2e-3 if run._DEVICE.type=='cuda' else 1e-6
torch.testing.assert_close(torch.tensor(result[0]),torch.tensor(run.predict_image(sample)),atol=tolerance,rtol=tolerance)
try:
    run.predict_image({**sample,'attribute_names':ATTRIBUTES[::-1]})
except ValueError:pass
else:raise AssertionError('Wrong order accepted')
print(json.dumps(result[:2]))
''')
    proc=subprocess.run([sys.executable,str(script),str(real.resolve())],cwd=tmp_path,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    p=np.asarray(json.loads(proc.stdout))
    assert p.shape==(2,40) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    target=np.tile(np.arange(40)%2,(2,1));target[1]=1-target[0]
    report=macro_score(p,target,np.ones_like(target,bool),['fixture']*2)
    assert np.isfinite(report['macro']['Challenge_Avg'])


def test_untrained_checkpoint_refused(tmp_path):
    checkpoint=tmp_path/'untrained.pt'
    torch.save({'trained':False},checkpoint)
    result=subprocess.run([sys.executable,'-m','upar.package','--checkpoint',str(checkpoint),
                           '--calibration',str(tmp_path/'missing.json'),'--output',str(tmp_path/'bad.zip')],
                           cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/'bad.zip').exists()
