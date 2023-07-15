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

sys.path.append("../")
from toolbox import (
    continous_and_generator,
    continous_or_generator,
    continous_xor_generator,
    forward_pass,
    set_seed,
    Config,
    clear_print,
    get_git_revision_hash,
)

np.printoptions(precision=3)

def show_summary(results, suptitle=None, ylim=(None, None)) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    results.xs(("loss", "train"), level=("stat", "mode"), axis=1).plot(
        figsize=(15, 5), legend=False, ax=ax[0], title="Loss"
    )
    results.xs(("accuracy", "train"), level=("stat", "mode"), axis=1).plot(
        figsize=(15, 5), legend=False, ax=ax[1], title="Accuracy"
    )
    results.xs(("f1", "train"), level=("stat", "mode"), axis=1).plot(
        figsize=(15, 5), legend=False, ax=ax[2], title="F1 score"
    )
    if suptitle is not None:
        plt.suptitle(f"Train summary {suptitle}")
    else:
        plt.suptitle("Train summary")
    if ylim[0] is not None and ylim[1] is not None:
        ax[0].set_ylim(ylim[0], ylim[1])
        ax[1].set_ylim(0, 1)
        ax[2].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

def load_results(path: str) -> pd.DataFrame:
    extension = os.path.splitext(path)[1]
    if extension == ".csv":
        results = pd.read_csv(path, index_col=[0], header=[0, 1, 2, 3, 4])
    elif extension == ".hdf":
        results = pd.read_hdf(path)
    return results
    
ylim = (0.0, 4.0)
results = load_results("results_1add73e8c0fe24a08ed58ec2df2ba0d457164e03.csv")
results = load_results("results_b25e7336d74e4d967c9b16600308c6a42decebcc_0.5.1.hdf")
results2 = load_results("results_b25e7336d74e4d967c9b16600308c6a42decebcc_0.6.2.hdf")
results_ANN = load_results("results_9e453ec29681a8fdfbddccfb3a722a81cfac76ce_0.6.2_ANN.hdf")
results3 = load_results("results_9e453ec29681a8fdfbddccfb3a722a81cfac76ce_0.6.2.hdf")

show_summary(results, "0.5.1", ylim)
show_summary(results2, "0.6.2", ylim)
show_summary(results_ANN, "0.6.2_ANN", ylim)
show_summary(results3, "0.6.2 100 epochs", ylim)


### runtimes boxplot
import pickle 
with open("runtimes.pkl", "rb") as f:
    run = pickle.load(f)

all_data = [run["AND"], run["OR"], run["XOR"]]
labels = ['AND', 'OR', 'XOR']

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('Rectangular box plot')

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1]:
    ax.yaxis.grid(True)
    ax.set_xlabel('replikowana bramka logiczna')
    ax.set_ylabel('czas treningu [s]')

plt.show()