import torch
import torch.nn as nn


def init_weights_xn(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
