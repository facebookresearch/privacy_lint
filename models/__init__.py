import torch.nn as nn
import torch.nn.functional as F
from .lenet import LeNet
from .lstmlm import LSTMLM

def build_model(params):
    if params.architecture == "lenet":
        return LeNet()
    elif params.architecture == "lstm":
        return LSTMLM(params)
