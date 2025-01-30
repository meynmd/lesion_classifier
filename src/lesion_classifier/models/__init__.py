from .resnet import *
from .resnext import *

model_factory = {
    'resnet18': build_resnet18,
    'resnext50_32x4d': build_resnext50
}
