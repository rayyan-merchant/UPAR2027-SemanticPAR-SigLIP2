"""DINOv3-specific spatial indexing, RoPE, gradients and checkpoint contract."""
import torch
import pytest
from dino.v3_model import DinoV3Backbone,Attention
from dino.model import pool_descriptors,preprocess
from PIL import Image


def test_v3_rope_matches_reference_formula_and_row_order():
    m=DinoV3Backbone();s,c=m.rope(20,10)
    periods=100.**(torch.arange(16,dtype=torch.float32)/16)
    for row,col in [(0,0),(0,9),(19,0),(19,9),(7,3)]:
        coords=torch.tensor([2*(row+.5)/20-1,2*(col+.5)/10-1])
        expected=(2*torch.pi*coords[:,None]/periods).flatten().tile(2)
        torch.testing.assert_close(s[row*10+col],expected.sin(),atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(c[row*10+col],expected.cos(),atol=2e-6,rtol=1e-5)
    assert s.shape==c.shape==(200,64)
    assert not any(k.endswith('qkv.bias') for k in m.state_dict())
    assert m.norm.eps==1e-5 and m.reg_token.shape==(1,4,768)


def test_v3_canvas_and_spatial_register_exclusion():
    x,box=preprocess(Image.new('RGB',(80,160)),canvas=(320,160))
    assert x.shape==(3,320,160) and box.tolist()==[0.,0.,160.,320.]
    t=torch.ones(1,205,2);t[:,0]=3;t[:,1:5]=999999
    d=pool_descriptors(t,box[None],grid=(20,10),patch=16)
    torch.testing.assert_close(d[:,0],torch.full((1,2),3.))
    torch.testing.assert_close(d[:,1:],torch.ones(1,5,2))


def test_v3_attention_gradient_and_invalid_canvas():
    a=Attention();x=torch.randn(1,7,768,requires_grad=True)
    sin=torch.zeros(2,64);cos=torch.ones(2,64)
    a(x,(sin,cos)).square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(a.qkv.weight.grad).all()
    m=DinoV3Backbone()
    with pytest.raises(ValueError):m(torch.zeros(1,3,319,160))
