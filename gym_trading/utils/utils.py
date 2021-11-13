import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

buys = [13103, 13108, 13113, 13118, 13123, 6123, 6128, 6133, 6138, 6143]

def plot_transactions(buys, sells):
    data = pd.read_csv("../../data_recorder/database/data_exports/demo_LTC-USD_20190926.csv")
    plot = data["midpoint"].plot(markevery=buys, marker='o', markerfacecolor='red')
    fig = plot.get_figure()
    fig.savefig("output.png")

plot_transactions(buys, [])