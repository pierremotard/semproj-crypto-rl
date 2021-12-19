import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_transactions(buys, sells, episode):
    data = pd.read_csv("XBTUSD_2020-01-03.csv")
    plot = data["midpoint"].plot(markevery=buys, marker='o', markerfacecolor='red')
    fig = plot.get_figure()
    fig.savefig("output_buys_{}.png".format(episode))

    plot = data["midpoint"].plot(markevery=sells, marker='x', markerfacecolor='green')
    fig = plot.get_figure()
    fig.savefig("output_sells_{}.png".format(episode))
