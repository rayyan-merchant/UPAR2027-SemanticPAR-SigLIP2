"""Offline DINOv2-registers ViT and body pooling.

Architecture follows Meta facebookresearch/dinov2 (Apache-2.0); changes:
minimal torch-only inference, aspect-preserving canvas and six-descriptor head.
Keep upstream LICENSE and attribution in every distributable package.
"""
import math
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__();self.heads=heads
        self.qkv=nn.Linear(dim,3*dim);self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,d))


class LayerScale(nn.Module):
    def __init__(self,dim):super().__init__();self.gamma=nn.Parameter(torch.ones(dim))
    def forward(self,x):return x*self.gamma


class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__();self.fc1=nn.Linear(dim,4*dim);self.fc2=nn.Linear(4*dim,dim)
    def forward(self,x):return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();self.norm1=nn.LayerNorm(dim,eps=1e-6);self.attn=Attention(dim,heads)
        self.ls1=LayerScale(dim);self.norm2=nn.LayerNorm(dim,eps=1e-6);self.mlp=MLP(dim);self.ls2=LayerScale(dim)
    def forward(self,x):
        x=x+self.ls1(self.attn(self.norm1(x)))
        return x+self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self,dim,patch):super().__init__();self.proj=nn.Conv2d(3,dim,patch,stride=patch)
    def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)


class DinoBackbone(nn.Module):
    def __init__(self,dim=768,depth=12,heads=12,patch=14,pretrain_grid=37,registers=4):
        super().__init__();self.dim=dim;self.patch=patch;self.registers=registers
        self.patch_embed=PatchEmbed(dim,patch)
        self.cls_token=nn.Parameter(torch.zeros(1,1,dim));self.mask_token=nn.Parameter(torch.zeros(1,dim))
        self.pos_embed=nn.Parameter(torch.zeros(1,1+pretrain_grid**2,dim))
        self.register_tokens=nn.Parameter(torch.zeros(1,registers,dim))
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)]);self.norm=nn.LayerNorm(dim,eps=1e-6)
    def position(self,h,w):
        g=int(math.sqrt(self.pos_embed.shape[1]-1))
        if h==w==g:return self.pos_embed
        p=self.pos_embed[:,1:].reshape(1,g,g,self.dim).permute(0,3,1,2)
        p=F.interpolate(p.float(),size=(h,w),mode='bicubic',align_corners=False,antialias=True).to(self.pos_embed.dtype)
        return torch.cat((self.pos_embed[:,:1],p.permute(0,2,3,1).reshape(1,h*w,self.dim)),dim=1)
    def forward(self,pixels):
        h,w=pixels.shape[-2]//self.patch,pixels.shape[-1]//self.patch
        x=self.patch_embed(pixels)
        # Keep pretrained FP32 CLS/position/residual behavior under autocast.
        x=torch.cat((self.cls_token.expand(x.shape[0],-1,-1),x),dim=1)+self.position(h,w)
        x=torch.cat((x[:,:1],self.register_tokens.expand(x.shape[0],-1,-1),x[:,1:]),dim=1)
        for block in self.blocks:x=block(x)
        return self.norm(x)


def preprocess(image,canvas=(322,168)):
    """RGB bicubic resize without distortion, centered ImageNet-mean padding."""
    image=image.convert('RGB');height,width=canvas
    scale=min(width/image.width,height/image.height)
    w=max(1,min(width,round(image.width*scale)));h=max(1,min(height,round(image.height*scale)))
    left=(width-w)//2;top=(height-h)//2
    rgb=np.asarray(image.resize((w,h),Image.Resampling.BICUBIC),dtype=np.float32)/255.
    mean=np.array([.485,.456,.406],dtype=np.float32);std=np.array([.229,.224,.225],dtype=np.float32)
    array=np.zeros((height,width,3),dtype=np.float32)
    array[top:top+h,left:left+w]=(rgb-mean)/std
    return torch.from_numpy(array.transpose(2,0,1).copy()),torch.tensor([left,top,w,h],dtype=torch.float32)


def pool_descriptors(tokens,boxes,grid=(23,12),patch=14,registers=4):
    """CLS, spatial mean, head, upper, lower, spatial max; never pool registers.

    Spatial means use overlap area with non-padding content. Body bands are
    fractions of the resized pedestrian extent, not precise pose estimates.
    """
    gh,gw=grid
    if tokens.shape[1]!=1+registers+gh*gw:raise ValueError('CLS/register/spatial token count mismatch')
    spatial=tokens[:,1+registers:].float();boxes=boxes.to(spatial.device)
    y=torch.arange(gh,device=spatial.device).repeat_interleave(gw).float()*patch
    x=torch.arange(gw,device=spatial.device).repeat(gh).float()*patch
    left,top,width,height=boxes.unbind(-1)
    x_overlap=(torch.minimum(x[None]+patch,(left+width)[:,None])-torch.maximum(x[None],left[:,None])).clamp(min=0)
    def weights(a,b):
        lo=top+a*height;hi=top+b*height
        overlap=(torch.minimum(y[None]+patch,hi[:,None])-torch.maximum(y[None],lo[:,None])).clamp(min=0)
        return overlap*x_overlap
    full=weights(0.,1.)
    if (full.sum(1)==0).any():raise ValueError('Empty content box')
    def mean(w):return (spatial*w[:,:,None]).sum(1)/w.sum(1).clamp(min=1e-12)[:,None]
    descriptors=[tokens[:,0].float(),mean(full),mean(weights(0.,.30)),mean(weights(.15,.65)),mean(weights(.50,1.))]
    descriptors.append(spatial.masked_fill((full==0)[:,:,None],-torch.inf).amax(1))
    return torch.stack(descriptors,dim=1)


class AttributeHead(nn.Module):
    def __init__(self,dim=768,attributes=40,dropout=.2):
        super().__init__();self.norm=nn.LayerNorm(dim)
        self.query=nn.Parameter(torch.empty(attributes,dim));self.classifier=nn.Parameter(torch.empty(attributes,dim))
        self.bias=nn.Parameter(torch.zeros(attributes));self.dropout=nn.Dropout(dropout)
        nn.init.normal_(self.query,std=.02);nn.init.normal_(self.classifier,std=.02)
    def forward(self,descriptors):
        if descriptors.ndim!=3 or descriptors.shape[1]!=6:raise ValueError('Expected six descriptors')
        z=self.norm(descriptors.float())
        attention=torch.softmax(torch.einsum('bkd,ad->bak',z,self.query)/math.sqrt(z.shape[-1]),dim=-1)
        pooled=torch.einsum('bak,bkd->bad',attention,z)
        return (self.dropout(pooled)*self.classifier[None]).sum(-1)+self.bias
