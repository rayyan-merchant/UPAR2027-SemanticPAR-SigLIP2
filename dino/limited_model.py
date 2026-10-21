"""Candidate B: official backbone plus matching source-only frozen head."""
import torch
from torch import nn
from dino.v3_api import DinoV3Backbone,AttributeHead,pool_descriptors

def choose_foundation_receipt(paths,expected_sha256):
    """Identical pinned copies in dataset/kernel outputs are not new models."""
    import json
    if not paths:raise ValueError('Pinned public timm foundation unavailable')
    for path in paths:
        receipt=json.loads(path.read_text())
        if receipt.get('repo')!='timm/vit_base_patch16_dinov3.lvd1689m' or receipt.get('revision')!='c6a5fb7d12bbd3cf3b0079253141c3332aaed7da' or receipt.get('actual_sha256')!=expected_sha256:
            raise ValueError('Conflicting pretrained provenance')
    preferred=[p for p in paths if 'upar2027-private-dinov3-foundation' in p.parts]
    return sorted(preferred or paths)[0]

def configure_trainable(backbone):
    backbone.requires_grad_(False)
    for block in backbone.blocks[-2:]:block.requires_grad_(True)
    backbone.norm.requires_grad_(True)
    allowed=('blocks.10.','blocks.11.','norm.')
    for name,p in backbone.named_parameters():
        if p.requires_grad != name.startswith(allowed):raise AssertionError('Unexpected trainable backbone parameter '+name)

class LimitedDINO(nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.backbone=backbone;self.head=head
        configure_trainable(self.backbone)
    def forward(self,pixels,boxes):
        return self.head(pool_descriptors(self.backbone(pixels),boxes))
    def optimizer(self):
        return torch.optim.AdamW([{'params':[p for p in self.backbone.parameters() if p.requires_grad],'lr':5e-6},{'params':self.head.parameters(),'lr':5e-5}],weight_decay=.01)
