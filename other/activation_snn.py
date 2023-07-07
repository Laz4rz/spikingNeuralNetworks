import snntorch as snn
import snntorch.functional as SF

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from toolbox import forward_pass
import seaborn as sns

sns.set_style("whitegrid")

# simple snn model 
net = nn.Sequential(
    # nn.Linear(1, 1, bias=False),
    snn.Leaky(beta=0.99, threshold=0.5, spike_grad=True, init_hidden=True, output=True)
)

# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.2)

def plot_neuron():
    n_neurons = 1
    x = torch.rand(n_neurons)
    x = torch.tensor([0.01])
    u = torch.zeros(n_neurons)
    leaky = snn.Leaky(beta=0.99, threshold=0.9, output=True)

    u_hist = [u]
    s_hist = [0]
    for i in range(1000):
        s, u = leaky(x, u)
        u_hist.append(u)
        s_hist.append(s)

    plt.plot(s_hist, marker="", color="red", label="wzbudzenie neuronu")
    plt.plot(u_hist, marker="", label="napięcie membrany")
    plt.xlim(0, 1000)
    plt.xlabel("krok czasu")
    plt.legend()

plot_neuron()
plt.show()

def plot_heaviside():
    x = np.arange(0, 2, 0.001)
    y = list(map(lambda x: 1 if x > 0.9 else 0, x))
    plt.plot(x, y)
    plt.ylabel("wzbudzenie neuronu")
    plt.xlabel("napięcie membrany")
    plt.xlim(0, 2)

plot_heaviside()
plt.show()

# final plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plot_heaviside()
plt.title("Funkcja aktywacji w sieci pulsującej \n dla progu pobudliwości neuronu $\\theta=0.9$")

plt.subplot(1, 2, 2)
plot_neuron()
plt.title("Zachowanie neuronu w sieci pulsującej \n dla stałego napięcia")

plt.suptitle("Sposób zachowania pojedynczego neuronu pulsującego", y=1.02)