import numpy as np
from dino.final_a import split_validation

def test_pixel_group_split_excludes_train_and_keeps_duplicates_together():
    hashes=['train0','train1']+['v'+str(i) for i in range(100)]+['train0','v12']
    index={'audited_pixel_sha256':hashes}
    s,c=split_validation(index,2)
    assert 102 not in s and 102 not in c
    assert set(s).isdisjoint(c)
    assert set(s)|set(c)==set(range(2,102))|{103}
    assert (14 in s)==(103 in s)
    assert set(hashes[i] for i in s).isdisjoint(hashes[i] for i in c)
    s2,c2=split_validation(index,2)
    assert np.array_equal(s,s2) and np.array_equal(c,c2)
