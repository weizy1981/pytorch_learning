import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

class PolyNet(nn.Module):
    def __init__(self):
        super(PolyNet, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out
