import torch.nn.functional as F
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, nonlin=F.relu):
        super().__init__()
        self.num_units = 100
        self.nonlin = nonlin

        self.dense0 = nn.Linear(input_size, 100)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(100, 10)
        self.output = nn.Linear(10, output_size)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X