#!/usr/local/bin/python3
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from constants import NUM_SIMS, ERROR_P, LEN_ERROR_P

def plot(x_vals, xlabel, name, br_time_vals, br_vars, sc_vals, sc_vars, sc_l1_vals=None, sc_l1_vars=None, log=False, bw=True):
    """ Plot the best response, iterations, social cost of equilibrium regression
    and social cost of L1 regression - all in one plot"""
    fig, ax1 = plt.subplots()
    color = 'red'
    #ax1.set_ylabel('BR Updates', color=color, fontsize=21)
    if log == True:
        ax1.semilogx(x_vals, br_time_vals, color=color, linewidth=4, label="BR updates", linestyle='dashed')
        ax1.fill_between(x_vals, br_time_vals-br_vars, br_time_vals+br_vars, color=color, alpha=0.25)
    else:
        ax1.plot(x_vals, br_time_vals, color=color, linewidth=4, label="BR updates", linestyle='dashed')
        ax1.fill_between(x_vals, br_time_vals-br_vars, br_time_vals+br_vars, color=color, alpha=0.25)
    ax1.tick_params(axis='x', labelsize=17)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=17)
    #ax1.legend() 
    
    color = 'blue'
    ax2 = ax1.twinx()

    ax2.set_ylabel('Price of Anarchy', color=color, fontsize=21)
    if log == True:
        ax2.semilogx(x_vals, sc_vals, color=color, linewidth=4, label="PPoA")
        ax2.fill_between(x_vals, sc_vals-sc_vars, sc_vals+sc_vars, color=color, alpha=0.25)
    else:
        ax2.plot(x_vals, sc_vals, color=color, linewidth=4, label="PPoA")
        ax2.fill_between(x_vals, sc_vals-sc_vars, sc_vals+sc_vars, color=color, alpha=0.25)
    
    if sc_l1_vals is not None:
        if log == True:
            ax2.semilogx(x_vals, sc_l1_vals, color=color, linestyle=(0, (1,1)), linewidth=4, label="PPoA (LAD Regression)")
        else:
            ax2.plot(x_vals, sc_l1_vals, color=color, linestyle=(0,(1,1)), linewidth=4, label="PPoA (LAD regression)")
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=17)
    plt.grid(False)
    #ax2.legend() 
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

def plot_over_p(x_vals, xlabel, name, sc_vals, sc_vars, sc_l1_vals=None, sc_l1_vars=None, log=False):
    """ Plot the best response, iterations, social cost of equilibrium regression
    and social cost of L1 regression - all in one plot"""
    fig, ax1 = plt.subplots()
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    ax1.set_xlabel('p', color='black', fontsize=20)
    ax1.set_ylabel('Price of Anarchy', color='black', fontsize=20)
    for i in range(LEN_ERROR_P):
        if log == True:
            ax1.semilogx(x_vals, sc_vals[i], color=colors[i], linewidth=3)
            ax1.fill_between(x_vals, sc_vals[i]-sc_vars[i], sc_vals[i]+sc_vars[i], color=colors[i], alpha=0.25)
        else:
            ax1.plot(x_vals, sc_vals[i], color=colors[i], linewidth=3)
            ax1.fill_between(x_vals, sc_vals[i]-sc_vars[i], sc_vals[i]+sc_vars[i], color=colors[i], alpha=0.25)
        
        if sc_l1_vals is not None:
            if log == True:
                ax1.semilogx(x_vals, sc_l1_vals[i], color=colors[i], linestyle=":", linewidth=4)
            else:
                ax1.plot(x_vals, sc_l1_vals[i], color=colors[i], linestyle=':', linewidth=4)
    
    legends = ['PPoA: q=1.2', 'PPoA: q=1.7', 'PPoA: q=2', 'PPoA: q=3', 'PPoA: q=5']
    ax1.legend(legends)
    ax1.tick_params(axis='x', labelsize=17)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=17)
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(name)

def read_from_csv(filename):
    csv_file = open(filename, mode='r')
    csv_reader = csv.reader(csv_file, delimiter=",")
    social_cost, social_cost_l1, br_iters, x_vals = np.array([]), np.array([]), np.array([]), np.array([])
    social_cost_var, social_cost_l1_var, br_iters_var = [], [], []
    
    social_costs, social_costs_l1, social_costs_var, social_costs_l1_var = [], [], [], []
    for i in range(LEN_ERROR_P):
        social_costs.append(np.array([]))
        social_costs_l1.append(np.array([]))
        social_costs_var.append(np.array([]))
        social_costs_l1_var.append(np.array([]))

    # CSV format is: 
    # <sweep param> <br iters> <var_br> <avg_sc> ... <avg_sc> <avg_sc_var> ... <avg_sc_var> <avg_sc_l1>...<avg_sc_l1>...<avg_sc_l1_var>...<avg_sc_l1_var>
    for row in csv_reader:
        # 95% confidence - scale of 1.96
        scale = 1.96
        
        x_vals = np.append(x_vals, float(row[0]))
        br_iters = np.append(br_iters, float(row[1]))
        br_iters_var = np.append(br_iters_var, float(row[2])/np.sqrt(NUM_SIMS)*scale)
        
        for i in range(LEN_ERROR_P):
            social_costs[i] = np.append(social_costs[i], float(row[3+i]))
        for i in range(LEN_ERROR_P):
            social_costs_var[i] = np.append(social_costs_var[i], float(row[3+LEN_ERROR_P+i])/np.sqrt(NUM_SIMS)*scale)
        for i in range(LEN_ERROR_P):
            social_costs_l1[i] = np.append(social_costs_l1[i], float(row[3+(2*LEN_ERROR_P)+i]))
        for i in range(LEN_ERROR_P):
            social_costs_l1_var[i] = np.append(social_costs_l1_var[i], float(row[3+(3*LEN_ERROR_P)+i])/np.sqrt(NUM_SIMS)*scale)
        
    return x_vals, br_iters, br_iters_var, social_costs, social_costs_var, social_costs_l1, social_costs_l1_var

if __name__ == "__main__":
    """ Run this file as "./plot_from_csv <csv_file> <x_label> <log true/false> <q_val true/false> <ppoa_plot true/false>"""
    filename = sys.argv[1]
    x_label = sys.argv[2]
    log_plot = sys.argv[3].lower() == 'true'
    # whether to plot social cost/ppoa with p=2 or for all the different p listed in constants 
    ppoa_p2 = True
    if len(sys.argv) >= 5:
        ppoa_p2 = sys.argv[4].lower() == 'true'
    # Whether to plot the plot the variations due to computing PPoA with different p values
    to_plot_over_p = False
    if len(sys.argv) >= 6:
        to_plot_over_p = sys.argv[5].lower() == 'true'

    assert((ppoa_p2 == True and to_plot_over_p == False) or \
            (ppoa_p2 == False and to_plot_over_p == True))

    x_vals, br_iters, br_iters_var, social_costs, social_costs_var, social_costs_l1, social_costs_l1_var = read_from_csv(filename)
    
    if ppoa_p2 == True:
        index = ERROR_P.index(2)
        social_costs, social_costs_var = social_costs[index], social_costs_var[index]
        social_costs_l1, social_costs_l1_var = social_costs_l1[index], social_costs_l1_var[index]

    if to_plot_over_p == True:
        plot_over_p(x_vals, x_label, x_label, social_costs, social_costs_var, log=log_plot)
    else:
        plot(x_vals, x_label, x_label, br_iters, br_iters_var, social_costs, social_costs_var, \
                social_costs_l1, social_costs_l1_var, log=log_plot)

    
    
    #plot(x_vals, x_label, x_label, br_iters, br_iters_var, social_costs[0], social_costs_var[0], social_costs_l1[0], social_costs_l1_var[0], log=log_plot)
    #plot_over_p(x_vals, x_label, x_label, social_costs, social_costs_var, log=log_plot)

