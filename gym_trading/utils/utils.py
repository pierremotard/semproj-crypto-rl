import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_transactions(buys, sells, net_worth_values, seen, datafile, episode):
    fig, ax = plt.subplots()
    data = pd.read_csv("data_recorder/database/data_exports/" + datafile, compression="xz")
    plot1 = data["midpoint"].plot(markevery=buys, marker='o', markerfacecolor='red')

    plot2 = data["midpoint"].plot(markevery=sells, marker='x', markerfacecolor='green')

    red_patch = mpatches.Patch(color='red', label='Buys')
    green_patch = mpatches.Patch(color='green', label='Sells')
    ax.legend(handles=[red_patch, green_patch])

    fig = plot2.get_figure()
    fig.savefig("output_buys_sells_{}.png".format(episode))

    plot = data["midpoint"].plot(markevery=seen, marker='o', markerfacecolor='green')
    fig = plot.get_figure()
    fig.savefig("output_seen_{}.png".format(episode))


    fig = plt.figure(figsize=(10,5))
    xAxis = range(len(net_worth_values))
    plt.plot(list(xAxis), net_worth_values, label = "plot net worth") 


    print("Starts at {}".format(min(seen)))
    print("Ends at {}".format(max(seen)))
  
    # Labeling the X-axis 
    plt.xlabel('Steps') 
    # Labeling the Y-axis 
    plt.ylabel('Net worth') 
    # Give a title to the graph
    plt.title('Plot net worth') 
    # Show a legend on the plot 
    plt.legend() 
    #Saving the plot as an image
    fig.savefig('net_worth_{}.jpg'.format(episode))


