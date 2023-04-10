import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import random

import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import snntorch.functional as SF
import snntorch.spikeplot as splt


print("I work!")

def and_generator(size: int):
  x = Tensor(np.random.choice([0, 1], (size, 2)))
  y = Tensor([1 if i[0] and i[1] else 0 for i in x]).reshape(size, 1)

  return list(zip(x, y))


def forward_pass(net, data, num_steps):
  spk_rec = []
  mem_hist = []
  utils.reset(net)

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      mem_hist.append(mem_out.cpu().detach().numpy())
      spk_rec.append(spk_out)

  return torch.stack(spk_rec), np.stack(mem_hist)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

sweep_config = {
    'method': 'grid',
    'name': 'bRuTeFoRcE_V2',
    'metric': {
        'goal': 'minimize',
        'name': 'train loss'
    },
    "description": "Lineaer(2, 8), Lineaer(8, 2) Leaky()",
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.1]
        },
        'batch_size': {
            'values': [16, 32, 128]
        },
        'beta': {
            'values': [0.5, 0.9]
        },
        'threshold': {
            'values': [0.5, 0.9]
        },
        'timesteps': {
            'values': [5, 10]
        },
        'epochs': {
            'values': [30]
        },
        'rates': {
            'values': [9, 7]
        },
        "surrogate": {
            "values": ["fast_sigmoid", "sigmoid", "straight_through_estimator"] # "triangular" doesnt work for some reason, spike rate works bad
         },
        'seed': {
            'values': [1, 2137, 69]
        }
    }
}
surrogate_grads = {
    "fast_sigmoid": surrogate.fast_sigmoid(),
    "triangular": surrogate.triangular(),
    "sigmoid": surrogate.sigmoid(),
    "straight_through_estimator": surrogate.straight_through_estimator(),
    "spike_rate_escape": surrogate.spike_rate_escape()
}
rates = {
   9: (0.9, 0.1), 
   7: (0.7, 0.3)
}
sweep_id = wandb.sweep(sweep_config, project='and-gate-snn')
device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
  wandb.init()
  config = wandb.config
  set_seed(config.seed)
  train_loader = DataLoader(and_generator(size=700), config.batch_size)
  test_loader = DataLoader(and_generator(size=300), config.batch_size)

  net = nn.Sequential(
      nn.Linear(2, 2),
      nn.Linear(8, 2),
      snn.Leaky(beta=config.beta, threshold=config.threshold, spike_grad=surrogate_grads[config.surrogate], init_hidden=True, output=True)
  )
  net = net.to(device)

  optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
  loss_fn = SF.mse_count_loss(correct_rate=rates[config.rates][0], ratesincorrect_rate=rates[config.rates][1])

  for epoch in range(config.epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.squeeze().to(device)

        net.train()
        spk_rec, mem_hist = forward_pass(net, data, config.timesteps) # forward-pass
        loss_val = loss_fn(spk_rec, targets) # loss calculation
        optimizer.zero_grad() # null gradients
        loss_val.backward() # calculate gradients
        optimizer.step() # update weights

    wandb.log({
    "epoch": epoch,
    "train loss": loss_val.item(),
    "accuracy": SF.accuracy_rate(spk_rec, targets)
    })

wandb.agent(sweep_id, train)
