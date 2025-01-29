import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .initialize import init_weights_xn

def build_resnet18(n_categories):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights, progress=False)
    model.fc = nn.Linear(
        512, 
        1, 
        bias=True
    )
    model.fc.apply(init_weights_xn)
    transforms = weights.transforms(antialias=True)

    return model, transforms
