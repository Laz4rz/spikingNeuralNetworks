import snntorch as snn
import snntorch.functional as SF

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from toolbox import forward_pass
import seaborn as sns

sns.set_style("whitegrid")

def plot_dirac():
    theta = 0.9
    def dirac(theta, x):
        return list(map(lambda x: 1 if abs((x-theta)) < 0.001 else 0, x))
    x = np.arange(-10, 10, 0.001)
    y = dirac(theta, x)
    plt.plot(x, y)

def plot_sigmoid_derivative():
    theta = 0.9
    def sigmoid_derivative(theta, x):
        top = np.exp(theta-x)
        bot = (top + 1)**2
        return top/bot
    x = np.arange(-10, 10, 0.001)
    y = sigmoid_derivative(theta, x)
    plt.plot(x, y)


def plot_sigmoid():
    theta = 0.9
    def sigmoid(x):
        return 1/(1 + np.exp(theta-x))
    x = np.arange(-10, 10, 0.001)
    y = sigmoid(x)
    plt.plot(x, y)


def plot_heaviside():
    x = np.arange(-10, 10, 0.001)
    y = list(map(lambda x: 1 if x > 0.9 else 0, x))
    plt.plot(x, y)
    plt.xlim(0, 2)

# final plot
def plot_final():
    plt.figure(figsize=(7, 7))

    plt.subplot(2, 2, 1)
    plt.title("Funkcja Heavisidea")
    plot_heaviside()

    plt.subplot(2, 2, 2)
    plt.title("Funkcja sigmoidalna")
    plot_sigmoid()

    plt.subplot(2, 2, 3)
    plt.title("Pochodna funkcji Heavisidea \n Delta Diraca")
    plot_dirac()

    plt.subplot(2, 2, 4)
    plt.title("Pochodna funkcji sigmoidalnej")
    plot_sigmoid_derivative()

    for ax in plt.gcf().get_axes():
        ax.set_ylim(-.01, 1.01)
        ax.set_xlim(0.9-3, 0.9+3)

plot_final()
plt.tight_layout()