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
import seaborn as sns
import wandb

import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import snntorch.functional as SF
import snntorch.spikeplot as splt

from toolbox import and_generator, or_generator, xor_generator, forward_pass, set_seed

sns.set(style="darkgrid")


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

def accuracy(spk_out, targets):
    with torch.no_grad():
        _, idx = spk_out.sum(dim=0).max(1)
        accuracy = ((targets == idx).float()).mean().item()
    return accuracy

def f1(spk_out, targets):
    with torch.no_grad():
        try:
            _, idx = spk_out.sum(dim=0).max(1)
            tp = ((idx == 1) & (targets == 1)).sum().item()
            fp = ((idx == 1) & (targets == 0)).sum().item()
            precision = tp / (tp + fp)

            fn = ((idx == 0) & (targets == 1)).sum().item()
            recall = tp / (tp + fn)

            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        except ZeroDivisionError:
            # print("ZeroDivisionError, f1 set to 0")
            return 0    

def predict_single(x, y, model, timesteps):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk, _ = forward_pass(model, torch.tensor([x, y], dtype=torch.float32).to(device), timesteps)
    _, idx = spk[:, None, :].sum(dim=0).max(1)
    return idx

def predict(data, model, timesteps):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if data.get_device() == -1:
        spk, _ = forward_pass(model, torch.tensor(data, dtype=torch.float32).to(device), timesteps)
    else:
        spk, _ = forward_pass(model, data, timesteps)
    _, idx = spk.sum(dim=0).max(1)
    return idx

def get_decision_surface(model, timesteps):
    xdata = np.linspace(0, 1, 10)
    ydata = np.linspace(0, 1, 10)
    X, Y, Z = [], [], []
    for x, y in np.array(list(itertools.product(xdata, ydata))):
        X.append(x)
        Y.append(y)
        Z.append(predict_single(x, y, model, timesteps).item())

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

def train(
    batch_size = batch_size,
    beta = beta,
    threshold = threshold,
    adam_betas = adam_betas,
    rates = rates,
    epochs = epochs,
    timesteps = timesteps,
    learning_rate = learning_rate,
    seed = seed
):
    wandb.init()
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

    for epoch in tqdm(range(epochs)):
        for i, (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.squeeze().to(device)

            net.train()
            spk_rec, mem_hist = forward_pass(net, data, timesteps) # forward-pass
            loss_val = loss_fn(spk_rec, targets) # loss calculation
            optimizer.zero_grad() # null gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights
            wandb.log({
                "epoch": epoch,
                "train loss": loss_val.item(),
                "train accuracy": accuracy(spk_rec, targets),
                "train f1": f1(spk_rec, targets)
                })

        for i, (data, targets) in enumerate(iter(test_loader)):
            data = data.to(device)
            targets = targets.squeeze().to(device)

            net.eval()
            spk_rec, mem_hist = forward_pass(net, data, timesteps)
            loss_val = loss_fn(spk_rec, targets)
            wandb.log({
                "epoch": epoch,
                "test loss": loss_val.item(),
                "test accuracy": accuracy(spk_rec, targets),
                "test f1": f1(spk_rec, targets)
                })

sweep_config = {
    'method': 'grid',
    'name': 'or_bRuTeFoRcE_V2_oneLayer_ANN',
    'metric': {
        'goal': 'minimize',
        'name': 'train loss'
    },
    "description": "Lineaer(2, 8), Leaky(), Lineaer(8, 2), Leaky() \n batch_size = 32 \n beta = 0.9 \n threshold = 0.9 \n surrogate_gradient = surrogate.fast_sigmoid() \n adam_betas = (0.9, 0.999) \n rates = (0.9, 0.1) \n epochs = 30 \n timesteps = 10 \n seed = 1 \n learning_rate = 1e-1",
    'parameters': {
        'seed': {
            'values': np.arange(0, 50, 1).tolist()
            }
        }
    }
sweep_id = wandb.sweep(sweep_config, project='seed-dependency')
wandb.agent(sweep_id, train)