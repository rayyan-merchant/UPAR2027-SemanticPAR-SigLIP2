import torch
from dino.v3_model import DinoV3Backbone
from dino.model import AttributeHead
from dino.limited_model import LimitedDINO

def test_identical_foundation_receipts_are_deduplicated_but_conflicts_fail(tmp_path):
    import json,pytest
    from dino.limited_model import choose_foundation_receipt
    dataset=tmp_path/'upar2027-private-dinov3-foundation';dataset.mkdir()
    frozen=tmp_path/'frozen_output';frozen.mkdir()
    receipt={'repo':'timm/vit_base_patch16_dinov3.lvd1689m','revision':'c6a5fb7d12bbd3cf3b0079253141c3332aaed7da','actual_sha256':'pinned'}
    paths=[frozen/'RECEIPT.json',dataset/'RECEIPT.json']
    for p in paths:p.write_text(json.dumps(receipt))
    assert choose_foundation_receipt(paths,'pinned')==dataset/'RECEIPT.json'
    paths[0].write_text(json.dumps({**receipt,'actual_sha256':'different'}))
    with pytest.raises(ValueError):choose_foundation_receipt(paths,'pinned')

def test_exact_last_two_blocks_and_norm_gradients():
    torch.set_num_threads(2);torch.manual_seed(2027)
    model=LimitedDINO(DinoV3Backbone(),AttributeHead())
    # Small canvas still uses real CLS/register/RoPE and all12 Transformer blocks.
    from dino.model import pool_descriptors
    image=torch.randn(1,3,32,32);tokens=model.backbone(image)
    desc=pool_descriptors(tokens,torch.tensor([[0.,0.,32.,32.]]),grid=(2,2),patch=16,registers=4)
    model.head(desc).square().mean().backward()
    for name,p in model.backbone.named_parameters():
        assert p.requires_grad==name.startswith(('blocks.10.','blocks.11.','norm.'))
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
        else:assert p.grad is None,name
    assert all(p.grad is not None for p in model.head.parameters())
    optimizer=model.optimizer();assert [g['lr'] for g in optimizer.param_groups]==[5e-6,5e-5]
