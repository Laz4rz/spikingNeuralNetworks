import snntorch as snn
import snntorch.functional as SF

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from toolbox import forward_pass

torch.set_printoptions(precision=3, sci_mode=False, linewidth=150)

def plot_spiking_history(spk, mem):
    mem = mem.squeeze()
    spk = spk.squeeze()
    n_neurons = mem.shape[1]
    n_steps = mem.shape[0]

    for n in range(n_neurons):
        plt.plot(mem[:, n])

    for step in range(n_steps):
        for n in range(n_neurons):
            if spk[step, n] == 1:
                plt.axvline(step, linestyle="--")

num_inputs = 2
num_hidden = 2
num_outputs = 2
num_steps = 100
beta = 0.99

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            print("\n\n\nstep:", step)
            cur1 = self.fc1(x)
            print("cur1 is \n", cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            print("spk1 is \n", spk1, "\nmem1 is \n", mem1)
            cur2 = self.fc2(spk1)
            print("cur2 is \n", cur2, "\nmem2 is \n", mem2)
            spk2, mem2 = self.lif2(cur2, mem2)
            print("spk2 is \n", spk2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    
net = Net()

for name, param in net.named_parameters():
    print(name, param.data)

with torch.no_grad():
    net.eval()
    x = torch.tensor([[0, 1]], dtype=torch.float32)
    spk, mem = net(x)

plot_spiking_history(spk, mem)
