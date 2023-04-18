import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import wandb

from sweep_config import sweep_config_ANN


def plot_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    x = Tensor(np.random.choice([0, 1], (700, 2)))
    y = Tensor([1 if i[0] and i[1] else 0 for i in x]).reshape(700, 1)
    pred = np.round(model(x).detach().numpy())
    cf_matrix = confusion_matrix(y.detach().numpy(), pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cf_matrix, annot=True, fmt='d')
    plt.title("Confusion matrix")
    plt.xlabel("prediction")
    plt.ylabel("ground truth")
    plt.show()


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


def and_generator(size: int):
  x = Tensor(np.random.choice([0, 1], (size, 2)))
  y = Tensor([1 if i[0] and i[1] else 0 for i in x]).reshape(size, 1)

  return list(zip(x, y))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.one_way = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.one_way(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train_sweep():
    wandb.init()
    config = wandb.config
    set_seed(config.seed)
    train_loader = DataLoader(and_generator(size=700), config.batch_size)

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    lloss = []
    for epoch in range(config.epochs):
        for batch, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss, current = loss.item(), batch * len(X)
        lloss.append(loss)
        acc = sum(np.round(pred.cpu().detach().numpy()) == Y.cpu().detach().numpy()) / len(Y)

        wandb.log({
        "epoch": epoch,
        "train loss": loss,
        "accuracy": acc
        })

def train(return_model=False, plot=False):
    train_loader = DataLoader(and_generator(size=700), 32)
    set_seed(0)
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=1) # Adam jednak nie jest stochastyczny, ok

    lloss = []
    for i in range(100):
        for batch, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Epoch: {i}, loss: {loss:>7f}")
            lloss.append(loss)
            acc = sum(np.round(pred.cpu().detach().numpy()) == Y.cpu().detach().numpy()) / len(Y)
    if plot:
        plt.plot(lloss)
    if return_model:
        return model

sweep_id = wandb.sweep(sweep_config_ANN, project='and-gate-snn')
wandb.agent(sweep_id, train_sweep)
