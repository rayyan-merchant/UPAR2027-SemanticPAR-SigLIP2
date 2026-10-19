"""Dependency-free SigLIP2-NaFlex vision model and semantic attribute head.

Compatible with PyTorch 2.4.1 inference. Vision submodule key names track the
Hugging Face SigLIP2 vision transformer for strict backbone weight import.
We use patch-level features instead of the original image-text pooling head.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class VisionConfig:
    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                 num_attention_heads=12, patch_size=16, num_patches=256, layer_norm_eps=1e-6):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.layer_norm_eps = layer_norm_eps

    def as_dict(self):
        return vars(self).copy()


class VisionEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = nn.Linear(3 * cfg.patch_size**2, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.num_patches, cfg.hidden_size)
        self._eval_pos_cache = {}
        self.grid_side = math.isqrt(cfg.num_patches)
        if self.grid_side ** 2 != cfg.num_patches:
            raise ValueError("Pretrained position table must be square")

    def train(self, mode=True):
        self._eval_pos_cache.clear()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self._eval_pos_cache.clear()
        return super()._apply(fn, recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._eval_pos_cache.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, pixels, spatial_shapes):
        patches = self.patch_embedding(pixels)
        pos = self.position_embedding.weight.reshape(self.grid_side, self.grid_side, -1)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        pieces = []
        for i in range(pixels.size(0)):
            h, w = (int(v) for v in spatial_shapes[i].tolist())
            if h * w > pixels.size(1):
                raise ValueError("spatial_shapes exceed patch count")
            cache_ok = not self.training and not torch.is_grad_enabled()
            key = (h, w, pixels.size(1), str(pos.device), str(patches.dtype))
            interp = self._eval_pos_cache.get(key) if cache_ok else None
            if interp is None:
                interp = F.interpolate(pos.float(), size=(h, w), mode="bilinear",
                                       align_corners=False, antialias=True)
                interp = interp.flatten(2).transpose(1, 2).reshape(h*w, -1)
                if h * w < pixels.size(1):
                    interp = torch.cat([interp, interp[:1].expand(pixels.size(1)-h*w, -1)], dim=0)
                interp = interp.to(dtype=patches.dtype)
                if cache_ok:
                    if len(self._eval_pos_cache) >= 64:
                        self._eval_pos_cache.clear()
                    self._eval_pos_cache[key] = interp.detach()
            pieces.append(interp)
        return patches + torch.stack(pieces, dim=0)


class VisionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, valid):
        b, n, dim = x.shape
        head_dim = dim // self.num_heads
        def transform(t):
            return t.reshape(b, n, self.num_heads, head_dim).transpose(1, 2)
        q = transform(self.q_proj(x))
        k = transform(self.k_proj(x))
        v = transform(self.v_proj(x))
        # SDPA boolean mask: True means a patch is valid and may be attended to.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :].bool(), dropout_p=0.)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, dim))


class VisionMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VisionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = VisionAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, valid):
        x = x + self.self_attn(self.layer_norm1(x), valid)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([VisionLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x, valid):
        for block in self.layers:
            x = block(x, valid)
        return x


class NativeSigLIP2Vision(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = VisionEmbeddings(cfg)
        self.encoder = VisionEncoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, pixels, valid, shapes):
        x = self.embeddings(pixels, shapes)
        x = self.encoder(x, valid)
        return self.post_layernorm(x)


class AttributeHead(nn.Module):
    def __init__(self, dim=768, nattr=40, heads=8, dropout=0.1):
        super().__init__()
        self.attribute_queries = nn.Parameter(torch.randn(nattr, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.)
        self.out_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.attr_weight = nn.Parameter(torch.randn(nattr, dim) * 0.02)
        self.attr_bias = nn.Parameter(torch.zeros(nattr))
        self.global_fc = nn.Linear(dim, nattr)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_tokens, valid):
        b = visual_tokens.size(0)
        queries = self.query_norm(self.attribute_queries)[None].expand(b, -1, -1)
        attn_out, _ = self.cross_attention(queries, self.patch_norm(visual_tokens),
                                           self.patch_norm(visual_tokens),
                                           key_padding_mask=~valid.bool(), need_weights=False)
        features = queries + attn_out
        features = self.out_norm(features + self.ffn(features))
        local_logits = (self.dropout(features) * self.attr_weight[None]).sum(-1) + self.attr_bias
        global_pool = (visual_tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return local_logits + self.global_fc(self.dropout(global_pool))


class UPARModel(nn.Module):
    def __init__(self, vision_cfg=None, nattr=40):
        super().__init__()
        self.cfg = vision_cfg or VisionConfig()
        self.vision_model = NativeSigLIP2Vision(self.cfg)
        self.head = AttributeHead(self.cfg.hidden_size, nattr=nattr,
                                  heads=min(8, self.cfg.num_attention_heads))

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
        tokens = self.vision_model(pixel_values, pixel_attention_mask, spatial_shapes)
        return self.head(tokens, pixel_attention_mask)

    def initialize_from_huggingface(self, hf_model):
        """Import model.vision_model state, refusing missing *backbone* tensors."""
        backbone = hf_model.vision_model
        source_cfg=hf_model.config.vision_config
        for key, value in self.cfg.as_dict().items():
            if getattr(source_cfg,key) != value:
                raise ValueError(f'Vision config mismatch {key}: native={value}, HF={getattr(source_cfg,key)}')
        if source_cfg.hidden_act != 'gelu_pytorch_tanh':
            raise ValueError(f'Unsupported pretrained activation: {source_cfg.hidden_act}')
        hf_state = backbone.state_dict()
        own_state = self.vision_model.state_dict()
        missing = [k for k in own_state if k not in hf_state]
        mismatched = [k for k in own_state if k in hf_state and own_state[k].shape != hf_state[k].shape]
        if missing or mismatched:
            raise RuntimeError(f"Cannot map HF backbone: missing={missing[:8]}, mismatched={mismatched[:8]}")
        self.vision_model.load_state_dict({k: hf_state[k] for k in own_state}, strict=True)

    def freeze_vision_except_last(self, n=4):
        for p in self.vision_model.parameters():
            p.requires_grad_(False)
        for block in (self.vision_model.encoder.layers[-n:] if n else []):
            for p in block.parameters():
                p.requires_grad_(True)
        for p in self.vision_model.post_layernorm.parameters():
            p.requires_grad_(n > 0)


@torch.no_grad()
def set_attribute_queries(model, text_embeddings):
    e = F.normalize(text_embeddings.float(), dim=-1)
    if e.shape != model.head.attribute_queries.shape:
        raise ValueError(f"Expected {tuple(model.head.attribute_queries.shape)}, got {tuple(e.shape)}")
    model.head.attribute_queries.copy_(e * math.sqrt(e.size(-1)) * 0.02)
