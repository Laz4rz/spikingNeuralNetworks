
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


def train_loop(net, train_loader, n_epochs, n_timesteps):
  loss_hist = [] 
  acc_hist = []
  spks = []
  for epoch in range(n_epochs):
      for i, (data, targets) in enumerate(iter(train_loader)):
          data = data.to(device)
          targets = targets.squeeze().to(device)

          net.train()
          spk_rec, mem_hist = forward_pass(net, data, n_timesteps) # forward-pass
          # return spk_rec, mem_hist
          spks.append(spk_rec.cpu().detach().numpy() )
          loss_val = loss_fn(spk_rec, targets) # loss calculation
          optimizer.zero_grad() # null gradients
          loss_val.backward() # calculate gradients
          optimizer.step() # update weights
          loss_hist.append(loss_val.item()) # store loss

          if i % 25 == 0:
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")
      
  return net, loss_hist, acc_hist, spks

def predict(x):
    net.eval()
    with torch.no_grad():
        res = forward_pass(net, Tensor(x).to(device), 10)[0]
    return res.sum(0).argmax().item()

np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(and_generator(size=700), 16)
test_loader = DataLoader(and_generator(size=300), 16)

net = nn.Sequential(
    nn.Linear(2, 8),
    nn.Linear(8, 2),
    snn.Leaky(beta=0.1, threshold=0.2, init_hidden=True, output=True)
)
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

net, loss_hist, acc_hist, spks = train_loop(net, train_loader, 30, 10)