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

import sys
sys.path.append('../')
from toolbox import continous_and_generator, continous_or_generator, continous_xor_generator, forward_pass, set_seed, Config, clear_print, get_git_revision_hash
np.printoptions(precision=3)

config = Config(
    batch_size=32,
    beta=0.9,
    threshold=0.9,
    surrogate_gradient=surrogate.fast_sigmoid(),
    adam_betas=(0.9, 0.999),
    rates=(0.9, 0.1),
    epochs=50,
    timesteps=10,
    data_seed=1,
    learning_rate=1e-2,
    model_seeds=[seed for seed in range(1, 51)]
)

def accuracy(spk_out, targets):
    with torch.no_grad():
        _, idx = spk_out.sum(dim=0).max(1)
        accuracy = ((targets == idx).float()).mean().item()
    return accuracy

def f1(spk_out, targets):
    from sklearn import metrics
    with torch.no_grad():
        _, idx = spk_out.sum(dim=0).max(1)
        f1 = metrics.f1_score(targets.cpu().numpy(), idx.cpu().numpy())
        return f1

def predict_single(x, y, model, timesteps):
    spk, _ = forward_pass(model, torch.tensor([x, y], dtype=torch.float32).to(device), timesteps)
    _, idx = spk[:, None, :].sum(dim=0).max(1)
    return idx

def predict(data, model, timesteps):
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

def create_results_df(config):
    # Create the index levels
    experiments = ["AND", "OR", "XOR"]
    seeds = config.model_seeds
    columns = ["stats"]
    stats = ["loss", "accuracy", "f1"]
    mode = ["train", "test"]

    # Create the multi-index
    multi_index = pd.MultiIndex.from_product(
        [experiments, seeds, columns, stats, mode], 
        names=["experiment", "model_seed", "column", "stat", "mode"]
    )

    # Create the DataFrame
    df = pd.DataFrame(index=range(config.epochs), columns=multi_index)
    return df

def update_results_df(df, experiment, model_seed, epoch, train_stats, test_stats):
    # Update the train stats
    for stat, value in train_stats.items():
        df.loc[epoch, (experiment, model_seed, "stats", stat, "train")] = value

    # Update the test stats
    for stat, value in test_stats.items():
        df.loc[epoch, (experiment, model_seed, "stats", stat, "test")] = value

# set seed for data creation
print("Data seed:", config.data_seed)
set_seed(config.data_seed)

dataloaders = (
    ("AND", DataLoader(continous_and_generator(size=700), config.batch_size), DataLoader(continous_and_generator(size=300), config.batch_size)),
    ("OR", DataLoader(continous_or_generator(size=700), config.batch_size), DataLoader(continous_or_generator(size=300), config.batch_size)),
    ("XOR", DataLoader(continous_xor_generator(size=700), config.batch_size), DataLoader(continous_xor_generator(size=300), config.batch_size))
)

results = create_results_df(config)

for name, train_loader, test_loader in dataloaders:
    print("Experiment:", name)
    for seed in config.model_seeds:
        print("Model seed:", seed)
        set_seed(seed=seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = nn.Sequential(
            nn.Linear(2, 8),
            snn.Leaky(beta=config.beta, threshold=config.threshold, spike_grad=config.surrogate_gradient, init_hidden=True),
            nn.Linear(8, 2),
            snn.Leaky(beta=config.beta, threshold=config.threshold, spike_grad=config.surrogate_gradient, init_hidden=True, output=True)
        ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=config.adam_betas)
        correct_rate, incorrect_rate = config.rates
        loss_fn = SF.mse_count_loss(correct_rate=correct_rate, incorrect_rate=incorrect_rate)
        
        for epoch in tqdm(range(config.epochs)):
            train_epoch_loss_val, train_epoch_acc_val, train_epoch_f1_val = 0, 0, 0
            for i, (data, targets) in enumerate(iter(train_loader)):
                data = data.to(device)
                targets = targets.squeeze().to(device)
                net.train()
                spk_rec, mem_hist = forward_pass(net, data, config.timesteps) # forward-pass
                loss_val = loss_fn(spk_rec, targets) # loss calculation
                optimizer.zero_grad() # null gradients
                loss_val.backward() # calculate gradients
                optimizer.step() # update weights
                train_epoch_loss_val += loss_val.item()
                train_epoch_acc_val += accuracy(spk_rec, targets)
                train_epoch_f1_val += f1(spk_rec, targets)

            train_epoch_loss_val = train_epoch_loss_val/len(train_loader)
            train_epoch_acc_val = train_epoch_acc_val/len(train_loader)
            train_epoch_f1_val = train_epoch_f1_val/len(train_loader)

            test_epoch_loss_val, test_epoch_acc_val, test_epoch_f1_val = 0, 0, 0
            for i, (data, targets) in enumerate(iter(test_loader)):
                data = data.to(device)
                targets = targets.squeeze().to(device)

                net.eval()
                spk_rec, mem_hist = forward_pass(net, data, config.timesteps)
                loss_val = loss_fn(spk_rec, targets)
                test_epoch_loss_val += loss_val.item()
                test_epoch_acc_val += accuracy(spk_rec, targets)
                test_epoch_f1_val += f1(spk_rec, targets)
            
            test_epoch_loss_val = test_epoch_loss_val/len(test_loader)
            test_epoch_acc_val = test_epoch_acc_val/len(test_loader)
            test_epoch_f1_val = test_epoch_f1_val/len(test_loader)

            update_results_df(
                results, 
                name, 
                seed, 
                epoch,
                {"loss": train_epoch_loss_val, "accuracy": train_epoch_acc_val, "f1": train_epoch_f1_val}, 
                {"loss": test_epoch_loss_val, "accuracy": test_epoch_acc_val, "f1": test_epoch_f1_val}
            )
    clear_print()

# end summary
fig, ax = plt.subplots(1, 3, figsize=(15, 15))
results.xs(("loss", "train"), level=("stat", "mode"), axis=1).plot(figsize=(15, 5), legend=False, ax=ax[0])
results.xs(("accuracy", "train"), level=("stat", "mode"), axis=1).plot(figsize=(15, 5), legend=False, ax=ax[1])
results.xs(("f1", "train"), level=("stat", "mode"), axis=1).plot(figsize=(15, 5), legend=False, ax=ax[2])
plt.suptitle("Train summary")
plt.tight_layout()
plt.show()

# dump results
git_hash = get_git_revision_hash()
results.to_csv(f"results_{git_hash}.csv")
