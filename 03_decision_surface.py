import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import snntorch.functional as SF
import snntorch.spikeplot as splt

from toolbox import and_generator, or_generator, xor_generator, forward_pass, set_seed

# @np.vectorize
# def predict(x, y, model, timesteps):
#     spk, _ = forward_pass(model, torch.tensor([x, y], dtype=torch.float32).to(device), timesteps)
#     _, idx = spk[:, None, :].sum(dim=0).max(1)
#     return idx

# def get_decision_surface(model, timesteps):
#     xdata = np.linspace(0, 1, 10)
#     ydata = np.linspace(0, 1, 10)
#     X, Y = np.meshgrid(xdata, ydata)
#     decision_surface = predict(X, Y, model, timesteps)
#     return decision_surface



def predict(x, y, model, timesteps):
    spk, _ = forward_pass(model, torch.tensor([x, y], dtype=torch.float32).to(device), timesteps)
    _, idx = spk[:, None, :].sum(dim=0).max(1)
    return idx

def get_decision_surface(model, timesteps):
    xdata = np.linspace(0, 1, 10)
    ydata = np.linspace(0, 1, 10)
    X, Y, Z = [], [], []
    for x, y in np.array(list(itertools.product(xdata, ydata))):
        X.append(x)
        Y.append(y)
        Z.append(predict(x, y, model, timesteps).item())

    X = np.array(X).reshape(10, 10)
    Y = np.array(Y).reshape(10, 10)
    Z = np.array(Z).reshape(10, 10)
    return Z

def plot_decision_surface(decision_surface):
    xdata = np.linspace(0, 1, 10)
    ydata = np.linspace(0, 1, 10)
    X, Y = [], []
    for x, y in np.array(list(itertools.product(xdata, ydata))):
        X.append(x)
        Y.append(y)
    X = np.array(X).reshape(10, 10)
    Y = np.array(Y).reshape(10, 10)
    plt.contourf(X, Y, decision_surface, cmap='plasma')
    plt.title("Decision surface", y=1.05)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def forward_pass(net, data, num_steps):
  spk_rec = []
  mem_hist = []
  utils.reset(net)

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      mem_hist.append(mem_out.cpu().detach().numpy())
      spk_rec.append(spk_out)

  return torch.stack(spk_rec), np.stack(mem_hist)

batch_size = 32
beta = 0.9
threshold = 0.9
surrogate_gradient = surrogate.fast_sigmoid()
adam_betas = (0.9, 0.999)
rates = (0.9, 0.1)
epochs = 30
timesteps = 10
seed = 1
learning_rate = 1e-1
seed = 3

set_seed(seed=seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(and_generator(size=700), 32)
test_loader = DataLoader(and_generator(size=300), 32)

net = nn.Sequential(
    nn.Linear(2, 8),
    snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate_gradient, init_hidden=True),
    nn.Linear(8, 2),
    snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate_gradient, init_hidden=True, output=True)
).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=adam_betas)
correct_rate, incorrect_rate = rates
loss_fn = SF.mse_count_loss(correct_rate=correct_rate, incorrect_rate=incorrect_rate)

surface_history = []
train_loss_history = []
test_loss_history = []
for epoch in tqdm(range(epochs)):
    with torch.no_grad():
        decision_surface = get_decision_surface(net, timesteps)
        surface_history.append(decision_surface)

    train_epoch_loss_val = 0
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.squeeze().to(device)

        net.train()
        spk_rec, mem_hist = forward_pass(net, data, timesteps) # forward-pass
        loss_val = loss_fn(spk_rec, targets) # loss calculation
        optimizer.zero_grad() # null gradients
        loss_val.backward() # calculate gradients
        optimizer.step() # update weights
        train_epoch_loss_val += loss_val.item()
    train_loss_history.append(train_epoch_loss_val/batch_size)


    test_epoch_loss_val = 0
    for i, (data, targets) in enumerate(iter(test_loader)):
        data = data.to(device)
        targets = targets.squeeze().to(device)

        net.eval()
        spk_rec, mem_hist = forward_pass(net, data, timesteps)
        loss_val = loss_fn(spk_rec, targets)
        test_epoch_loss_val += loss_val.item()
    test_loss_history.append(loss_val.item()/batch_size)



# plt.figure(figsize=(10, 10))
# Z = predict(X, Y)
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='plasma')
# plt.title("Continous prediction surface", y=1.05)
# plt.show()