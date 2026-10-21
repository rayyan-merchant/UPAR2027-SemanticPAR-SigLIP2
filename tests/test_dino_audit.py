import numpy as np
import pytest
from scripts.analyze_dinov3_frozen import domain

def test_independent_confusion_and_mask_hand_case():
 y=np.tile(np.array([1,0,1,0])[:,None],(1,40));p=np.tile(np.array([.9,.9,.1,.1])[:,None],(1,40));v=np.ones_like(y,bool);v[:,-1]=False
 m,a=domain(p,y,v,np.full(40,.5))
 assert m['mA']==.5 and m['Inst_Prec']==.25 and m['Inst_Rec']==.25 and m['Inst_F1']==.25
 assert m['Challenge_Avg']==pytest.approx(1/3) and m['Label_F1']==.5
 assert a[0]['tp']==a[0]['tn']==a[0]['fp']==a[0]['fn']==1 and a[-1]['mA'] is None

def test_mean_image_f1_is_distinct_from_harmonic_mean_pr():
 y=np.zeros((2,40));y[0,0]=1;y[1,:4]=1;p=np.zeros_like(y);p[0,:4]=1;p[1,0]=1
 m,_=domain(p,y,np.ones_like(y,bool),np.full(40,.5))
 assert m['Inst_F1']==.625 and m['mean_image_F1']==.4

def test_unsupported_labels_and_all_invalid_safe_zero():
 y=np.zeros((3,40));p=np.zeros_like(y)
 m,a=domain(p,y,np.ones_like(y,bool),np.full(40,.5))
 assert m['mA']==0 and m['all_valid_safe_zero_mA']==.5 and m['Challenge_Avg']==0
 assert all(x['mA'] is None for x in a)
 m,_=domain(p,y,np.zeros_like(y,bool),np.full(40,.5));assert all(x==0 for x in m.values())
