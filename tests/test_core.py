import numpy as np
import torch
from PIL import Image
from upar.preprocess import preprocess_image,target_size
from upar.model import UPARModel,VisionConfig
from upar.metrics import macro_score,calibrate_thresholds
from upar.attributes import ATTRIBUTES,PROMPTS
from upar.train import loss_weighted

def test_40_attrs():
    assert len(ATTRIBUTES)==40 and len(PROMPTS)==40 and len(set(ATTRIBUTES))==40

def test_preprocess_rectangular():
    for wh in [(50,100),(100,50),(71,83),(150,200)]:
        img=Image.new('RGB',wh,color='grey')
        p,v,s=preprocess_image(img,max_patches=16)
        assert p.shape==(16,768)
        assert v.sum()==s.prod()
        assert v.dtype==torch.bool

def test_native_model_forward():
    cfg=VisionConfig(hidden_size=48,intermediate_size=96,num_hidden_layers=2,num_attention_heads=4,
                     num_patches=16)
    m=UPARModel(cfg).eval()
    b=2
    x=torch.randn(b,16,768)
    v=torch.ones(b,16,dtype=torch.bool)
    shapes=torch.tensor([[4,4],[4,4]])
    with torch.no_grad(): out=m(x,v,shapes)
    assert out.shape==(2,40) and torch.isfinite(out).all()

def test_weighted_loss_invalid_labels():
    logits=torch.zeros(1,40)
    y=torch.zeros(1,40)
    v=torch.zeros(1,40,dtype=torch.bool)
    y[0,0]=-1
    v[0,1]=True
    wp=torch.ones(40);wn=torch.ones(40)
    assert torch.isclose(loss_weighted(logits,y,v,wp,wn),torch.tensor(.6931472),atol=1e-5)

def test_metric_domains():
    rng=np.random.default_rng(3)
    p=rng.random((12,40))
    y=rng.integers(0,2,(12,40)).astype('float32')
    v=np.ones((12,40),bool)
    domain=['A']*6+['B']*6
    m=macro_score(p,y,v,domain)
    assert len(m['domains'])==2 and 0<=m['macro']['Challenge_Avg']<=1
    t=calibrate_thresholds(p,y,v,domain,passes=1)
    assert t.shape==(40,) and np.isfinite(t).all()
