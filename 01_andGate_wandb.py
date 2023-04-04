
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

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


sweep_config = {
    'method': 'grid',
    'name': 'fighting snn loss',
    'metric': {
        'goal': 'minimize',
        'name': 'train loss'
    },
    "description": "Lineaer(2, 8), Linear(8, 2), Leaky(spike_grad=None)",
    'parameters': {
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01, 0.1]
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'beta': {
            'values': [0.1, 0.2, 0.5, 0.7, 0.9]
        },
        'threshold': {
            'values': [0.1, 0.2, 0.5, 0.7, 0.9]
        },
        'timesteps': {
            'values': [5, 10, 25, 50, 100]
        },
        'epochs': {
            'values': [10, 50, 100, 200, 300]
        },
        'rates': {
            'values': [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project='and-gate-snn')
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
  wandb.init()
  config = wandb.config
  train_loader = DataLoader(and_generator(size=700), config.batch_size)
  test_loader = DataLoader(and_generator(size=300), config.batch_size)

  net = nn.Sequential(
      nn.Linear(2, 8),
      nn.Linear(8, 2),
      snn.Leaky(beta=config.beta, threshold=config.threshold, init_hidden=True, output=True)
  )
  net = net.to(device)

  optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
  loss_fn = SF.mse_count_loss(correct_rate=config.rates[0], incorrect_rate=config.rates[1])

  for epoch in range(config.epochs):
      for i, (data, targets) in enumerate(iter(train_loader)):
          data = data.to(device)
          targets = targets.squeeze().to(device)

          net.train()
          spk_rec, mem_hist = forward_pass(net, data, config.timesteps) # forward-pass
          # return spk_rec, mem_hist
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
