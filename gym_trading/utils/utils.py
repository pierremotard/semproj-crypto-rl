import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_transactions(buys, sells, net_worth_values, seen, rewards, datafile, episode):
    fig, ax = plt.subplots(2,2, figsize=(18, 10))

    data = pd.read_csv("data_recorder/database/data_exports/" + datafile, compression="xz")
    ax[0][0].plot(data["midpoint"], markevery=buys, marker='o', markerfacecolor='red')

    ax[0][1].plot(data["midpoint"], markevery=sells, marker='o', markerfacecolor='green')


    # plot = data["midpoint"].plot(markevery=seen, marker='o', markerfacecolor='green')
    # fig = plot.get_figure()
    # fig.savefig("output_seen_{}.png".format(episode))

    print("Starts at {}".format(min(seen)))
    print("Ends at {}".format(max(seen)))

    xAxis = range(len(net_worth_values))
    ax[1][0].plot(list(xAxis), net_worth_values, label = "plot net worth") 
  
    # # Labeling the X-axis 
    # ax[0][0].xlabel('Steps') 
    # # Labeling the Y-axis 
    # ax1.ylabel('Net worth') 
    # # Give a title to the graph
    # ax1.title('Plot net worth') 
    # # Show a legend on the plot 
    # ax1.legend() 
    
    xAxis = range(len(rewards))
    ax[1][1].plot(list(xAxis), rewards, label = "plot rewards") 


    #Saving the plot as an image
    fig.savefig('buys_sells_net_worth_rewards_{}.jpg'.format(episode))

