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
