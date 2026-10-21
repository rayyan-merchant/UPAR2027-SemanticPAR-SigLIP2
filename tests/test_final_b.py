import ast
from pathlib import Path
import pytest
from dino.final_b import extension_allowed

def hist(values):return [{'source_selection_score':v} for v in values]

def test_source_only_optional_extension_boundaries():
    assert extension_allowed([],0)
    assert extension_allowed(hist([.8]*5),5)
    assert not extension_allowed(hist([.8,.81,.82,.83,.84,.839]),6)
    assert not extension_allowed(hist([.8,.81,.82,.83,.84,.84005]),6)
    assert extension_allowed(hist([.8,.81,.82,.83,.84,.841]),6)
    assert not extension_allowed(hist([.8,.81,.82,.83,.84,.841,.840]),7)
    assert extension_allowed(hist([.8,.81,.82,.83,.84,.841,.842]),7)
    assert not extension_allowed(hist([.8,.81,.82,.83,.84,.841,.842,.85]),8)

def test_final_model_rejects_lodo_initialization_before_loading():
    from dino.final_b import initial_model
    with pytest.raises(ValueError,match='not a LODO'):initial_model('PETA')

def test_final_modules_compile_and_have_no_hidden_target_evaluation():
    from dino import final_b,final_b_runner
    for module in [final_b,final_b_runner]:
        source=Path(module.__file__).read_text()
        ast.parse(source)
        assert 'sealed_target_evaluation' not in source

def test_npz_members_are_decompressed_once(monkeypatch):
    import numpy as np
    from dino import final_b_runner as runner
    calls=[]
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,name):calls.append(name);return np.arange(5)
    monkeypatch.setattr(runner.np,'load',lambda path:Archive())
    labels=runner.load_label_arrays('unused.npz')
    for i in range(100):
        for name in ['y','valid','domains']:_=labels[name][i%5]
    assert calls==['y','valid','domains']

def test_vectorized_annotation_audit_keeps_all_identity_checks():
    import numpy as np
    from dino.final_b_runner import validate_annotation_alignment
    labels={'y':np.array([[1.,0.],[0.,float('nan')]]),'valid':np.array([[True,True],[True,False]]),'domains':np.array(['PA100k','PETA'])}
    index={'paths':['PA100k/a.jpg','PETA/b.jpg']}
    rows=[(index['paths'][i],labels['y'][i].copy(),labels['valid'][i].copy(),labels['domains'][i]) for i in range(2)]
    validate_annotation_alignment(index,labels,rows,0)
    for field in ['y','valid','domains']:
        bad={k:v.copy() for k,v in labels.items()}
        if field=='y':bad[field][0,0]=0
        elif field=='valid':bad[field][0,0]=False
        else:bad[field][0]='PETA'
        with pytest.raises(ValueError):validate_annotation_alignment(index,bad,rows,0)
    with pytest.raises(ValueError):validate_annotation_alignment({'paths':list(reversed(index['paths']))},labels,rows,0)

def test_prediction_locator_includes_runs_and_requires_exact_hash(tmp_path):
    import hashlib
    from dino.final_b_runner import locate_pinned_siglip_predictions
    path=tmp_path/'original'/'runs'/'siglip2'/'best_val_predictions.npz'
    path.parent.mkdir(parents=True);path.write_bytes(b'pinned-original-predictions')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert locate_pinned_siglip_predictions(tmp_path,sha)==path
    with pytest.raises(ValueError):locate_pinned_siglip_predictions(tmp_path,'wrong-sha')
    path.parent.rename(path.parent.with_name('other_run'))
    assert locate_pinned_siglip_predictions(tmp_path,sha).parent.name=='other_run'
