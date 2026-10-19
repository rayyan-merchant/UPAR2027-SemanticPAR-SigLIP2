"""Real-image pretrained parity, with per-layer diagnostics and persisted evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import UPARModel, VisionConfig, set_attribute_queries
from .preprocess import preprocess_image
from .attributes import ATTRIBUTES, PROMPTS


def errors(a, b):
    a, b = a.detach().float(), b.detach().float()
    diff = (a-b).abs()
    return {'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
            'relative_l2': (torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()}


def code_digest():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in ['model.py','preprocess.py','attributes.py'])).hexdigest()


def check(images, foundation, output, max_patches=256, device='cpu'):
    from transformers import Siglip2Model, Siglip2ImageProcessor, AutoTokenizer
    import transformers
    if len(images) < 3:
        raise ValueError('At least three real images required for parity')
    torch.manual_seed(2027)
    hf=Siglip2Model.from_pretrained(foundation, attn_implementation='sdpa').eval().to(device)
    cfg=hf.config.vision_config
    native=UPARModel(VisionConfig(**{k:getattr(cfg,k) for k in VisionConfig().as_dict()})).eval().to(device)
    native.initialize_from_huggingface(hf)
    # Explicit slow NumPy/PIL reference avoids processor dispatch changing across versions.
    processor=Siglip2ImageProcessor.from_pretrained(foundation)
    tokenizer=AutoTokenizer.from_pretrained(foundation)
    token=tokenizer(PROMPTS,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
    token={k:v.to(device) for k,v in token.items()}
    with torch.inference_mode():
        text=hf.get_text_features(**token)
        direct=hf.text_model(**token).pooler_output
        text_error=errors(text,direct)
        set_attribute_queries(native,text)
    records=[]
    # FP32 strict enough to expose patch/activation/position errors. Relative L2
    # is stable when individual reference elements are close to zero.
    limits={'fp32':{'max_abs':.002,'mean_abs':.0002,'relative_l2':.0002},
            'fp16':{'max_abs':.08,'mean_abs':.008,'relative_l2':.008}}
    aspects=[]
    for name in images:
        with Image.open(name) as im:
            im=im.convert('RGB'); aspects.append(round(im.width/im.height,3))
            reference=processor(images=im,max_num_patches=max_patches,return_tensors='pt')
            px,mask,shape=preprocess_image(im,max_patches)
        diff=errors(reference['pixel_values'],px[None])
        same_mask=torch.equal(reference['pixel_attention_mask'].bool(),mask[None])
        same_shape=torch.equal(reference['spatial_shapes'],shape[None])
        preprocess_ok=diff['max_abs'] <= 1e-6 and same_mask and same_shape
        p=reference['pixel_values'].to(device); m=mask[None].to(device); s=shape[None]
        # Record every native block vs corresponding HF hidden-state output.
        with torch.inference_mode():
            # The internal HF transformer does not run the public wrapper's
            # hidden-state capture decorator. Use explicit hooks for diagnostics.
            reference_stages={}
            def capture(name):
                def hook(module, inputs, output):
                    reference_stages[name]=(output[0] if isinstance(output,(tuple,list)) else output).detach()
                return hook
            handles=[hf.vision_model.embeddings.register_forward_hook(capture('embeddings'))]
            handles.extend(block.register_forward_hook(capture(f'block_{index}'))
                           for index,block in enumerate(hf.vision_model.encoder.layers))
            try:
                ref=hf.vision_model(pixel_values=p,attention_mask=m,spatial_shapes=s.to(device))
            finally:
                for handle in handles:handle.remove()
            x=native.vision_model.embeddings(p,s)
            stages={'embeddings':errors(reference_stages['embeddings'],x)}
            for index,block in enumerate(native.vision_model.encoder.layers):
                x=block(x,m)
                stages[f'block_{index}']=errors(reference_stages[f'block_{index}'],x)
            last=native.vision_model.post_layernorm(x)
            stages['final']=errors(ref.last_hidden_state,last)
            own=native.vision_model(px[None].to(device),m,s)
            stages['end_to_end']=errors(ref.last_hidden_state,own)
        ok=preprocess_ok and all(all(v[k] <= limits['fp32'][k] for k in v) for v in stages.values())
        mixed=None
        if torch.device(device).type=='cuda':
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                mixed_ref=hf.vision_model(pixel_values=p,attention_mask=m,spatial_shapes=s.to(device)).last_hidden_state
                mixed_own=native.vision_model(px[None].to(device),m,s)
            mixed=errors(mixed_ref,mixed_own)
            ok=ok and all(mixed[k] <= limits['fp16'][k] for k in mixed)
        records.append({'image':str(name),'image_sha256':hashlib.sha256(Path(name).read_bytes()).hexdigest(),
                        'pixel_errors':diff,'mask_equal':same_mask,'shape_equal':same_shape,
                        'stages':stages,'amp_errors':mixed,'passed':ok})
    aspect_diversity=len(set(aspects))>=3
    result={'schema':1,'foundation':foundation,'resolved_revision':getattr(hf.config,'_commit_hash',None),
            'attributes':ATTRIBUTES,'prompts':PROMPTS,'vision_config':native.cfg.as_dict(),
            'code_sha256':code_digest(),'torch':torch.__version__,'transformers':transformers.__version__,
            'device':str(device),'max_patches':max_patches,'tolerances':limits,
            'text_errors':text_error,'query_shape':list(native.head.attribute_queries.shape),
            'distinct_aspect_ratios':len(set(aspects)),'images':records,
            'mixed_precision_checked':torch.device(device).type=='cuda',
            'passed':all(x['passed'] for x in records) and aspect_diversity and text_error['max_abs']<1e-6}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if result['passed']:
        from .checkpoint import atomic_save
        prepared=output.parent/'foundation_native.pt'
        atomic_save({'state_dict':native.cpu().state_dict(),'vision_config':native.cfg.as_dict(),
                     'max_patches':max_patches,'attributes':ATTRIBUTES,'trained':False,
                     'foundation':foundation,'parity_code_sha256':code_digest()},prepared)
        result['prepared_sha256']=hashlib.sha256(prepared.read_bytes()).hexdigest()
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    output.with_suffix('.md').write_text('# Pretrained parity report\n\n'+
        f"Passed: {result['passed']}; device: {device}; images: {len(records)}; distinct aspect ratios: {len(set(aspects))}.\n\n"+
        'Maximum/mean absolute and relative L2 errors for preprocessing, embeddings, all blocks, final features and text queries are in the adjacent JSON.\n'+
        f"Mixed precision tested: {result['mixed_precision_checked']}.\n",encoding='utf-8')
    if not result['passed']:
        raise AssertionError(f'Pretrained parity failed: {output}')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--foundation',default='google/siglip2-base-patch16-naflex')
    ap.add_argument('--image',action='append',required=True)
    ap.add_argument('--output',default='reports/PRETRAINED_PARITY.json')
    ap.add_argument('--max-patches',type=int,default=256)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    result=check(args.image,args.foundation,args.output,args.max_patches,args.device)
    print(json.dumps({'passed':result['passed'],'output':args.output}),flush=True)

if __name__=='__main__':main()
