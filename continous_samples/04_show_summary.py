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

def show_summary(results) -> None:
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
    plt.suptitle("Train summary")
    plt.tight_layout()
    plt.show()

def load_results(path: str) -> pd.DataFrame:
    extension = os.path.splitext(path)[1]
    if extension == ".csv":
        results = pd.read_csv(path, index_col=[0], header=[0, 1, 2, 3, 4])
    elif extension == ".hdf":
        results = pd.read_hdf(path)
    return results
    
results = load_results("results_1add73e8c0fe24a08ed58ec2df2ba0d457164e03.csv")
results = load_results("results_b25e7336d74e4d967c9b16600308c6a42decebcc.hdf")

show_summary(results)
