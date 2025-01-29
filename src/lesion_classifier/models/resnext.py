import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from .initialize import init_weights_xn

def build_resnext50(n_categories):
    weights = ResNeXt50_32X4D_Weights.DEFAULT
    model = resnext50_32x4d(weights=weights, progress=False)
    model.fc = nn.Linear(
        2048, 
        1, 
        bias=True
    )
    model.fc.apply(init_weights_xn)
    transforms = weights.transforms(antialias=True)

    return model, transforms
