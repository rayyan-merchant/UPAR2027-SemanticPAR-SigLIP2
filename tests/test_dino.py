import numpy as np
from PIL import Image
import torch
import pytest
from dino.model import DinoBackbone,AttributeHead,preprocess,pool_descriptors
from upar.train import loss_weighted

def test_token_shape_and_register_exclusion():
    model=DinoBackbone(dim=24,depth=1,heads=3,patch=14,pretrain_grid=2,registers=4)
    tokens=model(torch.randn(2,3,42,28));assert tokens.shape==(2,11,24)
    tokens=torch.zeros(1,11,1);tokens[:,0]=99;tokens[:,1:5]=9999
    tokens[:,5:,0]=torch.arange(6)
    d=pool_descriptors(tokens,torch.tensor([[0.,0.,28.,42.]]),grid=(3,2))
    assert d.shape==(1,6,1) and d[0,0,0]==99 and d[0,5,0]==5
    assert d[0,1,0]==2.5 and d[0,2,0]<d[0,4,0]

def test_letterbox_is_aspect_preserving_and_padding_zero():
    p,box=preprocess(Image.new('RGB',(64,128),(255,0,0)))
    assert p.shape==(3,322,168)
    left,top,w,h=box.tolist();assert w==161 and h==322 and abs(w/h-.5)<.001
    assert torch.equal(p[:,:,:int(left)],torch.zeros_like(p[:,:,:int(left)]))

def test_area_pool_matches_independent_pixel_overlap():
    tokens=torch.zeros(1,11,2);tokens[:,5:]=torch.arange(12).reshape(6,2)
    box=torch.tensor([[2.,3.,20.,34.]])
    actual=pool_descriptors(tokens,box,grid=(3,2))[0,1].numpy()
    weights=[]
    for row in range(3):
        for col in range(2):weights.append(max(0,min((row+1)*14,37)-max(row*14,3))*max(0,min((col+1)*14,22)-max(col*14,2)))
    expected=np.average(tokens[0,5:].numpy(),axis=0,weights=weights)
    np.testing.assert_allclose(actual,expected,rtol=1e-6)

def test_head_gradients_masking_and_order():
    torch.manual_seed(2027);head=AttributeHead(dim=24).eval();f=torch.randn(3,6,24)
    logits=head(f);assert logits.shape==(3,40)
    y=torch.zeros_like(logits);v=torch.ones_like(logits,dtype=torch.bool);v[:,7]=False
    y[:,7]=float('nan');loss=loss_weighted(logits,y,v,torch.ones(40),torch.ones(40));loss.backward()
    assert torch.isfinite(head.query.grad).all() and torch.isfinite(head.classifier.grad).all()
    before=head(f).detach()
    with torch.no_grad():head.bias[3]+=1
    after=head(f).detach();torch.testing.assert_close(after[:,3],before[:,3]+1)
    torch.testing.assert_close(after[:,[i for i in range(40) if i!=3]],before[:,[i for i in range(40) if i!=3]])

def test_wrong_token_layout_rejected():
    with pytest.raises(ValueError):pool_descriptors(torch.zeros(1,7,24),torch.tensor([[0.,0.,28.,28.]]),grid=(2,2))
