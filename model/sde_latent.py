import torch
import torch.nn as nn
import torchsde

# from https://github.com/google-research/torchsde/tree/master

# Time Integrator
def solve_sdefunc(odefunc, t, y0):
    ''' Solve odefunction using torchsde.sdeint() '''

    solution = torchsde.sdeint(odefunc, y0, t, rtol=1e-9, atol=1e-9)
    final_state = solution[-1]
    return final_state

class latent_SDE(nn.Module):
    def __init__(self, dim, latent_dim, SDE, t):
        super(latent_SDE, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(dim, latent_dim), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, dim))
        self.SDE = SDE
        self.t = t


    def forward(self, y):
        z = self.encoder(y)
        z = solve_sdefunc(self.SDE, self.t, z)
        z = self.decoder(z)
        return z

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, state_size, brownian_size=2):
        super().__init__()
        self.state_size = state_size
        self.brownian_size = brownian_size
        self.mu = nn.Sequential(nn.Linear(self.state_size, 
                                  128),
                                nn.GELU(),
                                nn.Linear(128, 
                                  self.state_size)) #512
        self.sigma = nn.Sequential(torch.nn.Linear(self.state_size, self.state_size * self.brownian_size))

    # Drift
    def f(self, t, y):
        return self.mu(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        batch_size = y.shape[0]
        return self.sigma(y).view(batch_size, 
                                  self.state_size, 
                                  self.brownian_size)


