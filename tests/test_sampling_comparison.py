from collections import Counter
import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset
from upar.data import domain_sampler, label_weights


class Rows(Dataset):
    def __init__(self, counts=(900,100)):
        self.rows=[(str(i),np.full(40,i%2,dtype=np.float32),np.ones(40,dtype=bool),d)
                   for d,n in zip(['large','small'],counts) for i in range(n)]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return i,self.rows[i][3]


def test_equal_domain_probability_not_equal_image_probability():
    ds=Rows();s=domain_sampler(ds,alpha=0.,generator=torch.Generator().manual_seed(2027))
    mass={d:sum(float(s.weights[i]) for i,r in enumerate(ds.rows) if r[3]==d) for d in ['large','small']}
    assert mass['large']==pytest.approx(mass['small'])
    assert float(s.weights[0])==pytest.approx(1/900)
    assert float(s.weights[-1])==pytest.approx(1/100)
    baseline=domain_sampler(ds)
    assert float(baseline.weights[0])*900/(float(baseline.weights[0])*900+float(baseline.weights[-1])*100)==pytest.approx(.75)


@pytest.mark.parametrize('alpha',[0.,.5])
def test_logging_preserves_exact_indices_and_rng(alpha):
    ds=Rows();g1=torch.Generator().manual_seed(2027);g2=torch.Generator().manual_seed(2027)
    regular=domain_sampler(ds,alpha=alpha,generator=g1)
    tracked=domain_sampler(ds,alpha=alpha,generator=g2,track_exposure=True)
    global_before=torch.get_rng_state().clone()
    for _ in range(3):
        a=list(regular);b=list(tracked)
        assert a==b and torch.equal(g1.get_state(),g2.get_state())
        e=tracked.exposure();assert e['draws']==dict(Counter(ds.rows[i][3] for i in b))
        assert e['unique_images_epoch']=={d:len({i for i in b if ds.rows[i][3]==d}) for d in ['large','small']}
    assert torch.equal(global_before,torch.get_rng_state())


def test_exposure_checkpoint_and_generator_resume():
    ds=Rows();g=torch.Generator().manual_seed(2027);s=domain_sampler(ds,alpha=0.,generator=g,track_exposure=True)
    list(s);state=s.exposure_state();rng=g.get_state().clone()
    expected=list(s);counts=s.exposure()
    restored_g=torch.Generator();restored_g.set_state(rng)
    resumed=domain_sampler(ds,alpha=0.,generator=restored_g,track_exposure=True);resumed.restore_exposure(state)
    assert list(resumed)==expected and resumed.exposure()==counts
    with pytest.raises(ValueError):resumed.restore_exposure({'large':[-1],'small':[]})


def test_actual_consumed_exposure_and_frequency_estimates_unchanged():
    ds=Rows();before=label_weights(ds)
    s=domain_sampler(ds,alpha=0.,generator=torch.Generator().manual_seed(2027),track_exposure=True)
    consumed=Counter()
    for _,domains in DataLoader(ds,batch_size=37,sampler=s,num_workers=0):consumed.update(domains)
    assert dict(consumed)==s.exposure()['draws']
    assert sum(consumed.values())==len(ds)
    assert abs(consumed['large']/len(ds)-.5)<.05
    after=label_weights(ds)
    for a,b in zip(before,after):assert np.array_equal(np.asarray(a),np.asarray(b))


def test_three_hour_comparison_cap_preserves_reserve():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kaggle"))
    from resource_policy import decide
    names=['PA100k','PETA','Market1501']
    expected={h:{'partitions':{k:{'rows':n} for k,n in [('source_train',20000),('source_selection',1000),('source_calibration',1000),('sealed_target_evaluation',1000)]}} for h in names}
    solo={h:{'images_per_second':140} for h in names};joint={h:{'images_per_second':130} for h in names[:2]}
    result=decide(expected,solo,joint,True,.1,12.5,max_session_hours=3,reserve_hours=8)
    assert result['allowed'] and result['session_cap_hours']==3 and result['final_training_reserve_hours']==8
