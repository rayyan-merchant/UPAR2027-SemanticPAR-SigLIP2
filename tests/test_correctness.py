import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from upar.attributes import ATTRIBUTES
from upar.data import load_rows,domain_sampler,label_weights
from upar.model import UPARModel,VisionConfig,set_attribute_queries
from upar.preprocess import preprocess_image
from upar.train import loss_weighted
from upar.metrics import macro_score,score_one_domain,attribute_report,calibrate_thresholds
from upar.checkpoint import atomic_save,training_payload,restore_rng


@pytest.fixture
def tiny():
    torch.set_num_threads(2)
    return UPARModel(VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                                 num_attention_heads=4,num_patches=16))


def test_label_parser_missing_and_nonfinite(tmp_path):
    path=tmp_path/'gt.csv'
    values=['0','1','-1','','nan','inf']+['0']*34
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        csv.writer(f).writerows([['# image']+ATTRIBUTES,['PETA/images/x.png']+values])
    rows=load_rows(path)
    assert rows[0][2][:6].tolist()==[True,True,False,False,False,False]
    assert rows[0][3]=='PETA'


def test_wrong_order_and_path_traversal(tmp_path):
    path=tmp_path/'gt.csv'
    with path.open('w',newline='') as f:
        csv.writer(f).writerows([['# image']+ATTRIBUTES,['../x.png']+['0']*40])
    with pytest.raises(ValueError,match='Unsafe'):load_rows(path)
    with path.open('w',newline='') as f:csv.writer(f).writerow(['# image']+ATTRIBUTES[::-1])
    with pytest.raises(ValueError,match='attributes'):load_rows(path)


def test_nan_loss_and_gradient():
    x=torch.zeros(1,40,requires_grad=True)
    y=torch.full((1,40),float('nan'));y[0,1]=1
    valid=torch.ones(1,40,dtype=torch.bool)
    loss=loss_weighted(x,y,valid,torch.ones(40),torch.ones(40))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad[0,0]==0 and x.grad[0,1]<0


def test_patch_rgb_interleaving():
    rgb=np.zeros((16,16,3),np.uint8);rgb[:]=[255,127,0]
    p,m,s=preprocess_image(Image.fromarray(rgb),max_patches=1)
    expected=torch.tensor([1.,127/255*2-1,-1.])
    assert torch.allclose(p[0,:3],expected,atol=1e-6)
    assert torch.allclose(p[0,3:6],expected,atol=1e-6)


def test_hf_processor_parity_patterned_images():
    from transformers import Siglip2ImageProcessor
    reference=Siglip2ImageProcessor()
    rng=np.random.default_rng(42)
    for h,w in [(67,29),(31,98),(91,73)]:
        im=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
        r=reference(images=im,max_num_patches=16,return_tensors='pt')
        p,m,s=preprocess_image(im,16)
        torch.testing.assert_close(r['pixel_values'][0],p,atol=1e-6,rtol=0)
        assert torch.equal(r['pixel_attention_mask'][0].bool(),m)
        assert torch.equal(r['spatial_shapes'][0],s)


def test_backward_queries_and_padding(tiny):
    tiny.eval()
    set_attribute_queries(tiny,torch.randn(40,32))
    assert tiny.head.attribute_queries.shape==(40,32)
    p=torch.randn(2,16,768);mask=torch.zeros(2,16,dtype=torch.bool);mask[:,:12]=True
    shapes=torch.tensor([[4,3],[3,4]])
    a=tiny(p,mask,shapes)
    changed=p.clone();changed[:,12:]=1000
    b=tiny(changed,mask,shapes)
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    a.sum().backward()
    assert tiny.head.attribute_queries.grad is not None
    assert torch.isfinite(tiny.head.attribute_queries.grad).all()


def test_head_only_freeze(tiny):
    tiny.freeze_vision_except_last(0)
    assert not any(p.requires_grad for p in tiny.vision_model.parameters())
    assert all(p.requires_grad for p in tiny.head.parameters())
    tiny.freeze_vision_except_last(1)
    assert all(p.requires_grad for p in tiny.vision_model.encoder.layers[-1].parameters())
    assert not any(p.requires_grad for p in tiny.vision_model.encoder.layers[0].parameters())


def test_sampler_mass_and_training_prevalence():
    rows=[]
    for domain,n in [('a',4),('b',16)]:
        rows.extend([('x',np.arange(40)%2,np.ones(40,bool),domain) for _ in range(n)])
    ds=SimpleNamespace(rows=rows,__len__=lambda:len(rows))
    # Special methods are looked up on the class.
    class Data:
        def __len__(self):return len(rows)
    ds=Data();ds.rows=rows
    sampler=domain_sampler(ds)
    w=np.asarray(sampler.weights)
    assert np.isclose(w[:4].sum()/w.sum(),1/3)
    wp,wn,p=label_weights(ds)
    np.testing.assert_allclose(p,np.arange(40)%2)
    np.testing.assert_allclose(wp,np.exp(1-p),rtol=1e-6)


def test_metrics_hand_computed_and_macro_equal_weight():
    y=np.tile([0,1],(2,20));v=np.ones_like(y,bool)
    perfect=np.array(y,float)
    good=score_one_domain(perfect,y,v,np.full(40,.5))
    # Each attribute needs positive and negative examples under current convention.
    y[1]=1-y[0];perfect=y.astype(float)
    good=score_one_domain(perfect,y,v,np.full(40,.5))
    assert good['mA']==1 and good['Inst_F1']==1 and good['Challenge_Avg']==1
    p=np.concatenate([perfect,np.tile(1-perfect,(3,1))])
    target=np.tile(y,(4,1));valid=np.ones_like(target,bool)
    r=macro_score(p,target,valid,['small']*2+['large']*6)
    assert r['macro']['Challenge_Avg']==.5
    tie=score_one_domain(np.full_like(perfect,.5),y,v,np.full(40,.5))
    assert tie['Inst_Prec']==0
    assert attribute_report(perfect,y,v)['f1']==[1.]*40


def test_calibration_improves_regularized_objective():
    y=np.tile(np.arange(40)%2,(8,1));y[4:]=1-y[4:]
    p=np.where(y==1,.4,.1);v=np.ones_like(y,bool);d=['a']*4+['b']*4
    t=calibrate_thresholds(p,y,v,d,passes=1)
    before=macro_score(p,y,v,d)['macro']['Challenge_Avg']
    after=macro_score(p,y,v,d,t)['macro']['Challenge_Avg']-.005*np.abs(t-.5).mean()
    assert after>=before


def test_checkpoint_exact_next_update(tmp_path,tiny):
    generator=torch.Generator().manual_seed(3)
    opt=torch.optim.AdamW(tiny.parameters(),lr=.001)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    p=torch.randn(1,16,768);mask=torch.ones(1,16,dtype=torch.bool);shape=torch.tensor([[4,4]])
    tiny.eval()
    tiny(p,mask,shape).sum().backward();opt.step();opt.zero_grad()
    path=tmp_path/'state.pt'
    atomic_save(training_payload(tiny,None,opt,scaler,generator,2,20,.7,{}),path)
    # Resume and compare one real Adam update, not just serialized metadata.
    other=UPARModel(tiny.cfg).eval();opt2=torch.optim.AdamW(other.parameters(),lr=.001)
    ck=torch.load(path,weights_only=False)
    other.load_state_dict(ck['state_dict']);opt2.load_state_dict(ck['optimizer'])
    for model,optimizer in [(tiny,opt),(other,opt2)]:
        model(p,mask,shape).sum().backward();optimizer.step()
    for a,b in zip(tiny.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    restore_rng(ck['rng'],generator)
    assert ck['epoch']==2 and ck['best_score']==.7 and not path.with_suffix('.pt.tmp').exists()
