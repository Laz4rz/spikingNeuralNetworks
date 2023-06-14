import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import seaborn as sns

sns.set_style("whitegrid")

# set data
data = [
    ("2018-06-11", "GPT-1", 0.12), 
    ("2018-10-31", "BERT-Large", 0.34), 
    ("2019-09-17", "Megatron-LM", 8.3),
    ("2019-11-05", "GPT-2", 1.5),  
    ("2020-02-24", "T5", 11), 
    ("2020-05-28", "GPT-3", 175), 
    # ("2021-10-01", "Megatron-Turing NLG", 530), # miesiac? 
    ("2022-04-02", "PaLM", 540),
]

brain = ("Human brain", 86)

date = [d[0] for d in data]
model = [d[1] for d in data]
params = [d[2] for d in data]

df = pd.DataFrame({"model": model, "params": params}, index=date)
df["days"] = pd.to_datetime(df.reset_index()["index"]).diff().fillna(pd.Timedelta("0D")).dt.days.values.cumsum()

# fit linear model
from sklearn.linear_model import LinearRegression

X = df["days"].values.reshape(-1, 1)
y = df["params"].values.reshape(-1, 1)

reg = LinearRegression(fit_intercept=False).fit(X, y)

# plot
plt.figure(figsize=(10, 5))
plt.title("Zmiana liczby parametrów sztucznych sieci neuronowych")
plt.ylabel("Liczba parametrów (mld)")
plt.xlabel("Data publikacji modelu")
plt.plot(df.index, df["params"], marker="s", markersize=5, markerfacecolor="red")

for i in data:
    date_moved_str = (pd.to_datetime(i[0]) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    print(i[0], date_moved_str)
    plt.annotate(i[1], xy=(i[0], i[2]), xytext=(i[0], i[2]+0.1*i[2]))


# plot log-linear model
plt.plot(df.index, reg.predict(X), linestyle="--", color="black")

plt.yscale("log")
# plt.xlim(df.index[0], df.index[-1])
plt.ylim(0.1, 1000)
plt.axhline(y=brain[1], color="r", linestyle="--")
plt.legend(["Liczba parametrów sztucznych sieci neuronowych", "dopasowany model liniowy", "Średnia liczba neuronów w ludzkim mózgu"])
plt.savefig("../Latex/n_params.png")
plt.show()

# plot annotation for each data point from dataframe

