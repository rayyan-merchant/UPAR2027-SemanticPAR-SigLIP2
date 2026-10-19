"""Atomic training-state checkpoints. Only load trusted local training artifacts."""
import os
import random
from pathlib import Path
import numpy as np
import torch


def atomic_save(payload, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'sampler':generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():raise RuntimeError('CUDA resume needs CUDA')
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['sampler'])


def training_payload(model, ema, optimizer, scaler, generator, epoch, step, best, config, **extra):
    return {'schema':2,'state_dict':model.state_dict(),
            'ema_state_dict':None if ema is None else ema.state_dict(),
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(generator),'epoch':epoch,'step':step,'best_score':best,
            'config':config,'vision_config':model.cfg.as_dict(),**extra}
