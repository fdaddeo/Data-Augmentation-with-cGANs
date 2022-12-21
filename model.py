import torch
import torch.nn as NN

class Net(NN.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x