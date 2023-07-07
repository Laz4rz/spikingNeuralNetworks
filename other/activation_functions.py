import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import seaborn as sns

sns.set_style("whitegrid")

x = np.linspace(-2, 2, 1000)

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_lim = (-1, 1)
x_lim = (-2, 2)

plt.figure(figsize=(15, 5))
plt.suptitle("Przykładowe funkcje aktywacji")

plt.subplot(1, 3, 1)
plt.plot(x, relu(x), label="ReLU")
plt.title("ReLU")
plt.ylim(y_lim)
plt.xlim(x_lim)

plt.subplot(1, 3, 2)
plt.plot(x, tanh(x), label="tanh")
plt.title("tanh")
plt.ylim(y_lim)
plt.xlim(x_lim)

plt.subplot(1, 3, 3)
plt.plot(x, sigmoid(x), label="sigmoid")
plt.title("sigmoid")
plt.ylim(y_lim)
plt.xlim(x_lim)

for ax in plt.gcf().axes:
    ax.set_xlabel("wejście")
    ax.set_ylabel("wyjście")