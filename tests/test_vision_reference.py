"""Architecture parity on random reference weights; real checkpoint gate is upar.parity."""
from types import SimpleNamespace
import torch
from transformers import Siglip2VisionConfig,Siglip2VisionModel
from upar.model import UPARModel,VisionConfig


def test_native_blocks_against_random_hf_reference():
    torch.set_num_threads(2);torch.manual_seed(123)
    config=Siglip2VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                              num_attention_heads=4,num_patches=16,patch_size=16)
    reference=Siglip2VisionModel(config).eval()
    native=UPARModel(VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                                 num_attention_heads=4,num_patches=16)).eval()
    native.initialize_from_huggingface(SimpleNamespace(vision_model=reference.vision_model,
                                      config=SimpleNamespace(vision_config=config)))
    p=torch.randn(3,16,768)
    shape=torch.tensor([[4,4],[4,3],[3,4]])
    m=torch.arange(16)[None]<shape.prod(1)[:,None]
    with torch.inference_mode():
        ref=reference(pixel_values=p,pixel_attention_mask=m,spatial_shapes=shape).last_hidden_state
        own=native.vision_model(p,m,shape)
    torch.testing.assert_close(ref,own,atol=2e-5,rtol=2e-5)


def test_autocast_preserves_reference_position_and_residual_precision():
    """CPU BF16 exposes the AMP position cast bug without requiring a GPU."""
    torch.set_num_threads(2);torch.manual_seed(123)
    config=Siglip2VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                              num_attention_heads=4,num_patches=16,patch_size=16)
    reference=Siglip2VisionModel(config).eval()
    native=UPARModel(VisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                                 num_attention_heads=4,num_patches=16)).eval()
    native.initialize_from_huggingface(SimpleNamespace(vision_model=reference.vision_model,
                                      config=SimpleNamespace(vision_config=config)))
    pixels=torch.randn(2,16,768)
    shapes=torch.tensor([[4,3],[3,4]])
    mask=torch.arange(16)[None]<shapes.prod(1)[:,None]
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
        expected=reference.vision_model.embeddings(pixels,shapes)
        actual=native.vision_model.embeddings(pixels,shapes)
        cached=native.vision_model.embeddings(pixels,shapes)
        ref=reference(pixel_values=pixels,pixel_attention_mask=mask,spatial_shapes=shapes).last_hidden_state
        own=native.vision_model(pixels,mask,shapes)
    assert expected.dtype==actual.dtype==cached.dtype==torch.float32
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    torch.testing.assert_close(expected,cached,atol=0,rtol=0)
    torch.testing.assert_close(ref,own,atol=2e-5,rtol=2e-5)
