import torch
import torchsde
from matplotlib.pyplot import *
        
def simulate_1d_rw(nsteps=1000, p=0.5, stepsize=1):
    steps = [ 1*stepsize if torch.rand(1) < p else -1*stepsize for i in range(nsteps) ]
    y = torch.cumsum(torch.tensor(steps),dim=0)
    x = list(range(len(y)))

    return x, list(y)

simulation_data = {}
nsims = 5
for i in range(nsims):
    x, y = simulate_1d_rw()
    simulation_data['x'] = x
    simulation_data['y{col}'.format(col=i)] = y

plot(simulation_data['y1'])
savefig("plot.png")

