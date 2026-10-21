"""Torch-only DINOv3 ViT-B/16 inference matching timm1.0.24 Eva.
Architecture adapted from timm (Apache2.0); pretrained weights DINOv3 license.
Keep both licenses and provenance in distributions. FP32 RoPE periods are the
public timm checkpoint convention, not the original Meta BF16-truncated periods.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv2d(3,768,16,stride=16)
    def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(768,2304,bias=False);self.proj=nn.Linear(768,768)
    def forward(self,x,rope):
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,12,64).permute(2,0,3,1,4).unbind(0)
        sin,cos=rope
        def rotate(z):
            spatial=z[:,:,5:]
            a,d=spatial.chunk(2,-1)
            spatial=spatial*cos+torch.cat((-d,a),dim=-1)*sin
            return torch.cat((z[:,:,:5],spatial),dim=2).type_as(v)
        q,k=rotate(q),rotate(k)
        return self.proj(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,c))

class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(768,3072);self.fc2=nn.Linear(3072,768)
    def forward(self,x):return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(768,eps=1e-5);self.attn=Attention()
        self.gamma_1=nn.Parameter(torch.ones(768));self.norm2=nn.LayerNorm(768,eps=1e-5)
        self.mlp=MLP();self.gamma_2=nn.Parameter(torch.ones(768))
    def forward(self,x,rope):
        x=x+self.gamma_1*self.attn(self.norm1(x),rope)
        return x+self.gamma_2*self.mlp(self.norm2(x))

class DinoV3Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.patch_embed=PatchEmbed()
        self.cls_token=nn.Parameter(torch.zeros(1,1,768));self.reg_token=nn.Parameter(torch.zeros(1,4,768))
        self.blocks=nn.ModuleList([Block() for _ in range(12)]);self.norm=nn.LayerNorm(768,eps=1e-5)
        self.register_buffer('periods',100.**(2.*torch.arange(16,dtype=torch.float32)/32),persistent=False)
    def rope(self,h,w):
        # CPU-generated normalized coordinates match timm before moving to device.
        y=torch.arange(.5,h,dtype=torch.float32)/float(h)
        x=torch.arange(.5,w,dtype=torch.float32)/float(w)
        coords=torch.stack(torch.meshgrid(y,x,indexing='ij'),dim=-1).flatten(0,1)
        coords=(2.*coords-1.)[:,:,None].to(device=self.periods.device,dtype=torch.float32)
        angles=(2*math.pi*coords/self.periods.float()[None,None,:]).flatten(1).tile(2)
        return angles.sin(),angles.cos()
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or x.shape[-2]%16 or x.shape[-1]%16:
            raise ValueError('Require RGB canvas divisible by patch16')
        h,w=x.shape[-2]//16,x.shape[-1]//16
        rope=self.rope(h,w);z=self.patch_embed(x)
        z=torch.cat((self.cls_token.expand(len(x),-1,-1),self.reg_token.expand(len(x),-1,-1),z),dim=1)
        for block in self.blocks:z=block(z,rope)
        return self.norm(z)
