import numpy as np
import torch
from dino.ensemble_submission import blend_and_calibrate,prepare_sample
from upar.attributes import ATTRIBUTES
from PIL import Image

def test_raw_probability_mixture_then_source_threshold_shift():
    s=torch.linspace(-6,6,80).reshape(2,40);d=-s/2;t=torch.linspace(.22,.78,40)
    shift=torch.logit(t);out=blend_and_calibrate(s,d,shift)
    raw=.75*torch.sigmoid(s)+.25*torch.sigmoid(d)
    expected=torch.sigmoid(torch.logit(raw)-shift)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    assert torch.equal(out>.5,raw>t)
    wrong=.75*torch.sigmoid(s-shift)+.25*torch.sigmoid(d-shift)
    assert (out-wrong).abs().max()>.01

def test_exact_attribute_order_and_shared_image_decode(tmp_path):
    path=tmp_path/'image.png';Image.new('RGB',(45,101),(220,50,25)).save(path)
    p,m,shape,image,box=prepare_sample({'image_path':str(path),'attribute_names':ATTRIBUTES})
    assert p.shape==(256,768) and m.shape==(256,) and shape.prod()<=256
    assert image.shape==(3,320,160) and box.shape==(4,)
    import pytest
    with pytest.raises(ValueError):prepare_sample({'image_path':str(path),'attribute_names':ATTRIBUTES[::-1]})
