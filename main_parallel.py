# Sweep values for n, p, xdim and run simulations for them

import numpy as np
from best_response_algo import best_response_algo
from p_regression import generate_data, fit_model
from plot_from_csv import plot
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm 

import time
import concurrent.futures 

## default values ##
P = 2
N = 100
NUM_SIMS = 1000
ALPHA = 1
XDIM = 5
NUM_CORES = 12

def run_simulation(tup):
    "Find the nash equilbrium for the given input and get the necessary stats"""

    X, true_Y, n_, with_l1, p = tup
    proj_Y = np.matmul(X, fit_model(X, true_Y))
    MSE_h = np.sum(np.power(proj_Y - true_Y, 2))

    if with_l1:
        proj_Y_l1 = np.matmul(X, fit_model(X, true_Y, p=1))
        MSE_l1 = np.sum(np.power(proj_Y_l1 - true_Y, 2))
    else:
        MSE_l1 = 1

    equi_exists, counter, iters, equi_report, beta_equi = best_response_algo(X, true_Y, p)
    social_cost, social_cost_l1, br_iters = 1, 1, 1
    if equi_exists:
        proj = np.matmul(X, beta_equi)
        MSE_eq = np.sum(np.power(proj - true_Y, 2))
     
        if MSE_eq != 0 and MSE_h != 0:
            social_cost = MSE_eq/MSE_h
            social_cost_l1 = MSE_l1/MSE_h
        br_iters = iters
    else:
        print("NO EQUILIBRIUM")
    return (br_iters, social_cost, social_cost_l1)

def sweep_n(n_vals, with_l1=False, to_plot=True):
    """ Sweep the number of agents, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""

    csv_file = open("sweep_n.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []
    social_cost_var, social_cost_l1_var, br_iters_var = [], [], []

    for n_ in n_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        n_ = int(n_)
        print("n = ", n_)

        inputs = []
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=n_)
            tup = (X, true_Y, n_, with_l1, P)
            inputs.append(tup)
        
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for future in futures:
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_.append(sc)
            social_cost_l1_.append(sc_l1)

        br_iters_, social_cost_, social_cost_l1_ = np.array(br_iters_), np.array(social_cost_), np.array(social_cost_l1_)
        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_), np.sqrt(np.var(social_cost_))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_), np.sqrt(np.var(social_cost_l1_))
        
        csv_writer.writerow([str(n_), str(avg_br), str(var_br), \
                str(avg_sc), str(var_sc), str(avg_sc_l1), str(var_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var.append(var_sc)
        social_cost_l1_var.append(var_sc_l1)

    if to_plot:
        if with_l1 == True:
            plot(n_vals, "# of agents", "sweep_n", br_iters, social_cost, social_cost_l1)
        else:
            plot(n_vals, "# of agents", "sweep_n", br_iters, social_cost)

def sweep_p(p_vals, with_l1=False, to_plot=True):
    """ Sweep the p-norm, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""
    
    csv_file = open("sweep_p.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []
    social_cost_var, social_cost_l1_var, br_iters_var, equi_found = [], [], [], []

    for p_ in p_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        print("p = ", p_)
        
        inputs = []
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=N)
            tup = (X, true_Y, N, with_l1, p_)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for future in futures:
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_.append(sc)
            social_cost_l1_.append(sc_l1)
        
        br_iters_, social_cost_, social_cost_l1_ = np.array(br_iters_), np.array(social_cost_), np.array(social_cost_l1_)
        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_), np.sqrt(np.var(social_cost_))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_), np.sqrt(np.var(social_cost_l1_))
        
        csv_writer.writerow([str(p_), str(avg_br), str(var_br), \
                str(avg_sc), str(var_sc), str(avg_sc_l1), str(var_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var.append(var_sc)
        social_cost_l1_var.append(var_sc_l1)

    if to_plot:
        if with_l1 == True:
            plot(p_vals, "p value", "sweep_p", br_iters, social_cost, social_cost_l1)
        else:
            plot(p_vals, "p value", "sweep_p", br_iters, social_cost)

def sweep_d(d_vals, with_l1=False, to_plot=True):
    """ Sweep the dimension of the X, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""
    
    csv_file = open("sweep_d.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []
    social_cost_var, social_cost_l1_var, br_iters_var = [], [], []

    for d_ in d_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        d_ = int(d_)
        print("d = ", d_)
        
        inputs = []
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=d_, n=N)
            tup = (X, true_Y, N, with_l1, P)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for future in futures:
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_.append(sc)
            social_cost_l1_.append(sc_l1)

        br_iters_, social_cost_, social_cost_l1_ = np.array(br_iters_), np.array(social_cost_), np.array(social_cost_l1_)
        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_), np.sqrt(np.var(social_cost_))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_), np.sqrt(np.var(social_cost_l1_))
        
        csv_writer.writerow([str(d_), str(avg_br), str(var_br), \
                str(avg_sc), str(var_sc), str(avg_sc_l1), str(var_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var.append(var_sc)
        social_cost_l1_var.append(var_sc_l1)
    
    if to_plot:
        if with_l1 == True:
            plot(d_vals, "X dimension", "sweep_d", br_iters, social_cost, social_cost_l1)
        else:
            plot(d_vals, "X dimension", "sweep_d", br_iters, social_cost)

def benchmark_n():
    print("Benchmarking n")
    start = time.time()
    sweep_n([10, 20, 30, 40], with_l1=True, to_plot=True)
    end = time.time()
    print(end-start)

def benchmark_d():
    print("Benchmarking d")
    start = time.time()
    sweep_d([1,3,5,7], with_l1=True, to_plot=False)
    end = time.time()
    print(end-start)

def benchmark_p():
    print("Benchmarking p")
    start = time.time()
    sweep_p([1.2,3], with_l1=True, to_plot=False)
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    n_vals = np.concatenate([np.arange(10,100,10), np.arange(100,1000,100), np.arange(1000, 11000, 1000)])
    sweep_n(n_vals, with_l1=True)

    d_vals = np.arange(1, 20, 2)
    sweep_d(d_vals, with_l1=True)
    
    #p_vals = [1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10, 100]
    #sweep_p(p_vals, with_l1=True)
   



