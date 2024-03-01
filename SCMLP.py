import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Dropout, ModuleList
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims = [64, 16, 16], dropout_rate = 0.5, n_classes = 14, input_size = 2000):
        super(MLP, self).__init__()
        self.dims = [input_size, *dims]
        self.dropout = dropout_rate
        self.layers = ModuleList()

        for i in range(len(self.dims) - 1):
            self.layers.append(Sequential(
                Linear(self.dims[i], self.dims[i + 1]),
                LeakyReLU(),
                Dropout(self.dropout)))
        self.last = Linear(self.dims[-1], n_classes)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return self.last(x)
