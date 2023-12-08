import torch
import torch.nn as nn
import torchdiffeq




class ODE(nn.Module):

    def __init__(self, dim):
        super(ODE, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(), #SiLU
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, dim)
        )
        # self.t = torch.linspace(0, 0.01, 2)

    def forward(self, t, y):
        # y = y.reshape(-1, 1)
        res = self.net(y)
        return res