import snntorch as snn
import snntorch.functional as SF

import torch
import torch.nn as nn

import numpy as np

from toolbox import forward_pass


# simple snn model 
net = nn.Sequential(
    nn.Linear(2, 2, bias=False),
    snn.Leaky(beta=0.99, threshold=0.5, spike_grad=True, init_hidden=True),
    nn.Linear(2, 2, bias=False),
    snn.Leaky(beta=0.99, threshold=0.5, spike_grad=True, init_hidden=True, output=True)
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.2)

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=torch.float32)
y_pred = forward_pass(net, x, 10)

loss = loss_fn(y_pred, y)
