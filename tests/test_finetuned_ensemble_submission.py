import torch
from types import SimpleNamespace
from dino import ensemble_finetuned_submission as api
from upar.attributes import ATTRIBUTES

def test_raw_blend_uses_new_global_threshold_only():
    s=torch.linspace(-5,5,80).reshape(2,40);d=-s/3;t=torch.linspace(.22,.78,40)
    raw=.75*s.sigmoid()+.25*d.sigmoid()
    expected=torch.sigmoid(torch.logit(raw)-torch.logit(t))
    torch.testing.assert_close(api.blend_and_calibrate(s,d,torch.logit(t)),expected,atol=0,rtol=0)

def test_finetuned_descriptors_are_not_quantized_to_frozen_cache_dtype(monkeypatch):
    seen=[]
    def head(x):
        seen.append(x)
        return torch.zeros(len(x),40)
    monkeypatch.setattr(api,'_DEVICE',torch.device('cpu'))
    monkeypatch.setattr(api,'_SIGLIP',lambda p,m,s:torch.zeros(len(p),40))
    monkeypatch.setattr(api,'_DINO',lambda p:torch.zeros(len(p),205,768))
    monkeypatch.setattr(api,'_HEAD',head)
    monkeypatch.setattr(api,'_LOGIT_THRESHOLD',torch.zeros(40))
    monkeypatch.setattr(api,'_POOL',SimpleNamespace(map=lambda fn,items:map(fn,items)))
    monkeypatch.setattr(api,'prepare_sample',lambda x:(torch.zeros(1,768),torch.ones(1,dtype=torch.bool),torch.tensor([1,1]),torch.zeros(3,320,160),torch.tensor([0.,0.,160.,320.])))
    monkeypatch.setattr(api,'pool_descriptors',lambda tokens,boxes:torch.full((len(tokens),6,768),1.0001))
    out=api.predict_batch([{'image_path':'unused','attribute_names':ATTRIBUTES}])
    assert len(out)==1 and len(out[0])==40
    assert seen[0].dtype==torch.float32 and float(seen[0][0,0,0])>1.00009

def test_official_attribute_order_is_enforced_before_decoding():
    import pytest
    with pytest.raises(ValueError,match='attribute order'):
        api.prepare_sample({'image_path':'unused','attribute_names':ATTRIBUTES[::-1]})
