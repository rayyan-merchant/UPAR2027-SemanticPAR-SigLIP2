"""DINOv3 uses the preserved six-descriptor/head implementation at patch16."""
from dino.v3_model import DinoV3Backbone
from dino.model import AttributeHead,preprocess as _preprocess,pool_descriptors as _pool

def preprocess(image):return _preprocess(image,canvas=(320,160))
def pool_descriptors(tokens,boxes):return _pool(tokens,boxes,grid=(20,10),patch=16,registers=4)
