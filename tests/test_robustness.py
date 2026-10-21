import random
import numpy as np
import torch
from PIL import Image
from upar.robustness import train_augment,install
from upar import preprocess

def test_robustness_bounds_and_deterministic_rng():
    image=Image.fromarray(np.arange(96*48*3,dtype=np.uint8).reshape(96,48,3))
    for seed in range(20):
        random.seed(seed);a=train_augment(image)
        random.seed(seed);b=train_augment(image)
        assert a.size==image.size and a.mode=='RGB'
        assert np.array_equal(a,b) and np.isfinite(np.asarray(a)).all()

def test_explicit_install_leaves_evaluation_preprocessing_identical():
    image=Image.new('RGB',(41,95),(230,35,90));before=preprocess.preprocess_image(image)
    original=preprocess.train_augment
    try:
        install();after=preprocess.preprocess_image(image,augment=False)
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        assert preprocess.train_augment is train_augment
    finally:preprocess.train_augment=original
