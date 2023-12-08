import torch
import torch.nn as nn
import torchdiffeq

# Time Integrator
def solve_odefunc(odefunc, t, y0):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]
    return final_state


class latent_ODE(nn.Module):
    def __init__(self, dim, latent_dim, ODE, t):
        super(latent_ODE, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(dim, latent_dim), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, dim))
        self.ODE = ODE
        self.t = t


    def forward(self, y):
        z = self.encoder(y)
        z = solve_odefunc(self.ODE, self.t, z)
        z = self.decoder(z)
        return z




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