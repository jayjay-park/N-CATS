import torch
import torch.nn as nn
import torchdiffeq
import sys
import torchsde

# sys.path.append('..')
# from model.ode import *

# functions
def solve_odefunc(odefunc, t, y0, logvar, device):
    ''' Solve odefunction using torchdiffeq.odeint() '''

    solution = torchdiffeq.odeint(odefunc, y0, t, rtol=1e-9, atol=1e-9, method="rk4")
    final_state = solution[-1]

    return final_state



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



class nvade(nn.Module):

    def __init__(self, t, device, input_dim=784, hidden_dim=400, latent_dim=200, encoded_dim = 2):
        super(nvade, self).__init__()
        self.t = t
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
            )

        # self.diffusion = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.Sigmoid(),
            # nn.Linear(hidden_dim, latent_dim),
            # nn.Sigmoid(),
            # nn.Linear(latent_dim, latent_dim),
            # nn.Sigmoid()
            # )
        
        # latent mean and variance 
        self.mean_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
            nn.Linear(latent_dim, encoded_dim))
        self.logvar_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
            nn.Linear(latent_dim, encoded_dim))

        # ODE
        self.ode = ODE(encoded_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
            )


    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        # print("logvar encode:", logvar)
        return mean, logvar, x

    def reparameterization(self, mean, logvar, device):
        # add brownian noise
        # print(logvar.shape): [7546, 4] or [3234, 4]

        bm = torchsde.BrownianInterval(t0=0., t1=1., size=logvar.shape)
        dW = bm(0., 0.1).to(device)

        # epsilon = torch.randn_like(logvar).to(self.device)  
  
        z = mean + logvar * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    # forward
    def forward(self, x):

        mean, logvar, x_ = self.encode(x)
        z = self.reparameterization(mean, logvar, self.device)
        z = solve_odefunc(self.ode, self.t, z, logvar, self.device)
        x_hat = self.decode(z)
        return x_hat, mean, logvar