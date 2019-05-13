#!/usr/bin/python3
import csv
import sys
import matplotlib.pyplot as plt


def plot(x_vals, xlabel, name, br_time_vals, sc_vals, sc_l1_vals=None, log=False):
    """ Plot the best response, iterations, social cost of equilibrium regression
    and social cost of L1 regression - all in one plot"""

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('BR Iterations', color=color)
    if log == True:
        ax1.semilogx(x_vals, br_time_vals, color=color)
    else:
        ax1.plot(x_vals, br_time_vals, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Social Cost', color=color)
    if log == True:
        ax2.semilogx(x_vals, sc_vals, color=color)
    else:
        ax2.plot(x_vals, sc_vals, color=color)
    
    if sc_l1_vals is not None:
        if log == True:
            ax2.semilogx(x_vals, sc_l1_vals, color=color, linestyle=":")
        else:
            ax2.plot(x_vals, sc_l1_vals, color=color, linestyle=':')
    
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(name)
    #plt.show()

def read_from_csv(filename):
    csv_file = open(filename, mode='r')
    csv_reader = csv.reader(csv_file, delimiter=",")
    social_cost, social_cost_l1, br_iters, x_vals = [], [], [], []
   
    # CSV format is: 
    # <sweep param> <br iters> <var_br> <avg_sc> <avg_sc_var> <avg_sc_l1> <avg_sc_l1_var>
    for row in csv_reader:
        x_vals.append(float(row[0]))
        br_iters.append(float(row[1]))
        social_cost.append(float(row[3]))
        social_cost_l1.append(float(row[5]))
        
    return x_vals, br_iters, social_cost, social_cost_l1

if __name__ == "__main__":
    """ Run this file as "./plot_from_csv <csv_file> <x_label> <log true/false> """
    filename = sys.argv[1]
    x_label = sys.argv[2]
    log_plot = sys.argv[3].lower() == 'true'

    x_vals, br_iters, social_cost, social_cost_l1 = read_from_csv(filename)
    plot(x_vals, x_label, x_label, br_iters, social_cost, social_cost_l1, log_plot)

