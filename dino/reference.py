"""Official HF to original Meta tensor mapping, reference/testing only."""
import torch

def convert(hf):
    state = {}
    for a, b in [('embeddings.cls_token', 'cls_token'), ('embeddings.mask_token', 'mask_token'), ('embeddings.position_embeddings', 'pos_embed'), ('embeddings.register_tokens', 'register_tokens'), ('embeddings.patch_embeddings.projection.weight', 'patch_embed.proj.weight'), ('embeddings.patch_embeddings.projection.bias', 'patch_embed.proj.bias'), ('layernorm.weight', 'norm.weight'), ('layernorm.bias', 'norm.bias')]:
        state[b] = hf[a]
    for i in range(12):
        a = f'encoder.layer.{i}.'
        b = f'blocks.{i}.'
        for kind in ['weight', 'bias']:
            state[b + 'attn.qkv.' + kind] = torch.cat([hf[a + 'attention.attention.' + x + '.' + kind] for x in ['query', 'key', 'value']])
            state[b + 'attn.proj.' + kind] = hf[a + 'attention.output.dense.' + kind]
            for x in ['norm1', 'norm2', 'mlp.fc1', 'mlp.fc2']:
                state[b + x + '.' + kind] = hf[a + x + '.' + kind]
        for n in [1, 2]:
            state[b + f'ls{n}.gamma'] = hf[a + f'layer_scale{n}.lambda1']
    return state
