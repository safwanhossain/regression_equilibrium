# Sweep values for n, p, xdim and run simulations for them

import numpy as np
from best_response_algo import best_response_algo
from p_regression import generate_data, fit_model
#from plot_from_csv import plot
import csv
from constants import P, N, NUM_SIMS, ALPHA, XDIM, NUM_CORES, ERROR_P

import time
import concurrent.futures 

def run_simulation(tup):
    "Find the nash equilbrium for the given input and get the necessary stats"""
    a = None
    if len(tup) == 5:
        X, true_Y, n_, with_l1, p = tup
    if len(tup) == 6:
        X, true_Y, n_, with_l1, p, a = tup

    proj_Y = np.matmul(X, fit_model(X, true_Y))
    errors_honest = [np.sum(np.power(np.abs(proj_Y - true_Y), k)) for k in ERROR_P]
    
    if with_l1:
        proj_Y_l1 = np.matmul(X, fit_model(X, true_Y, p=1))
        errors_l1 = [np.sum(np.power(np.abs(proj_Y_l1 - true_Y), k)) for k in ERROR_P]
    else:
        errors_l1 = [1 for k in ERROR_P]

    equi_exists, counter, iters, equi_report, beta_equi = best_response_algo(X, true_Y, p, test=True, alpha=a)
    social_cost, social_cost_l1, br_iters = 1, 1, 1
    if equi_exists:
        proj = np.matmul(X, beta_equi)
        errors_equi = [np.sum(np.power(np.abs(proj - true_Y), k)) for k in ERROR_P]
     
        if errors_equi[1] != 0 and errors_equi[1] != 0:
            social_cost = [errors_equi[k]/errors_honest[k] for k in range(len(ERROR_P))]
            social_cost_l1 = [errors_l1[k]/errors_honest[k] for k in range(len(ERROR_P))]
        br_iters = iters
    else:
        print("NO EQUILIBRIUM")
    return (br_iters, social_cost, social_cost_l1)

def sweep_n(n_vals, with_l1=False):
    """ Sweep the number of agents, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""

    csv_file = open("sweep_n.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = np.ones((len(n_vals), len(ERROR_P))), np.ones((len(n_vals), len(ERROR_P))), [], []
    social_cost_var, social_cost_l1_var, br_iters_var = np.ones((len(n_vals), len(ERROR_P))), np.ones((len(n_vals), len(ERROR_P))), []

    for index, n_ in enumerate(n_vals):
        social_cost_, social_cost_l1_, br_iters_ = np.ones((NUM_SIMS, len(ERROR_P))), np.ones((NUM_SIMS, len(ERROR_P))), []
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

        for sim, future in enumerate(futures):
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_[sim, :] = sc
            social_cost_l1_[sim, :] = sc_l1

        br_iters_ = np.array(br_iters_)
        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_, axis=0), np.sqrt(np.var(social_cost_, axis=0))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_, axis=0), np.sqrt(np.var(social_cost_l1_, axis=0))
        
        avg_sc_str = [str(avg_sc[k]) for k in range(len(ERROR_P))]
        var_sc_str = [str(var_sc[k]) for k in range(len(ERROR_P))]
        avg_sc_l1_str = [str(avg_sc_l1[k]) for k in range(len(ERROR_P))]
        var_sc_l1_str = [str(var_sc_l1[k]) for k in range(len(ERROR_P))]
        row = [str(n_)] + [str(avg_br)] + [str(var_br)] + avg_sc_str + var_sc_str + avg_sc_l1_str + var_sc_l1_str
        csv_writer.writerow(row)

        csv_file.flush()

        br_iters.append(avg_br)
        social_cost[index, :] = avg_sc
        social_cost_l1[index, :] = avg_sc_l1
        equi_found.append( social_cost_.shape[0] / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var[index, :] = var_sc
        social_cost_l1_var[index, :] = var_sc_l1

def sweep_p(p_vals, with_l1=False):
    """ Sweep the p-norm, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""
    
    csv_file = open("sweep_p.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = np.ones((len(p_vals), len(ERROR_P))), np.ones((len(p_vals), len(ERROR_P))), [], []
    social_cost_var, social_cost_l1_var, br_iters_var = np.ones((len(p_vals), len(ERROR_P))), np.ones((len(p_vals), len(ERROR_P))), []

    for index, p_ in enumerate(p_vals):
        social_cost_, social_cost_l1_, br_iters_ = np.ones((NUM_SIMS, len(ERROR_P))), np.ones((NUM_SIMS, len(ERROR_P))), []
        print("p = ", p_)
        
        inputs = []
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=N)
            tup = (X, true_Y, N, with_l1, p_)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for sim, future in enumerate(futures):
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_[sim, :] = sc
            social_cost_l1_[sim, :] = sc_l1
        
        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_, axis=0), np.sqrt(np.var(social_cost_, axis=0))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_, axis=0), np.sqrt(np.var(social_cost_l1_, axis=0))
        
        avg_sc_str = [str(avg_sc[k]) for k in range(len(ERROR_P))]
        var_sc_str = [str(var_sc[k]) for k in range(len(ERROR_P))]
        avg_sc_l1_str = [str(avg_sc_l1[k]) for k in range(len(ERROR_P))]
        var_sc_l1_str = [str(var_sc_l1[k]) for k in range(len(ERROR_P))]
        row = [str(p_)] + [str(avg_br)] + [str(var_br)] + avg_sc_str + var_sc_str + avg_sc_l1_str + var_sc_l1_str
        csv_writer.writerow(row)
        
        csv_file.flush()

        br_iters.append(avg_br)
        social_cost[index, :] = avg_sc
        social_cost_l1[index, :] = avg_sc_l1
        equi_found.append( social_cost_.shape[0] / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var[index, :] = var_sc
        social_cost_l1_var[index, :] = var_sc_l1

def sweep_d(d_vals, with_l1=False):
    """ Sweep the dimension of the X, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""
    
    csv_file = open("sweep_d.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = np.ones((len(d_vals), len(ERROR_P))), np.ones((len(d_vals), len(ERROR_P))), [], []
    social_cost_var, social_cost_l1_var, br_iters_var = np.ones((len(d_vals), len(ERROR_P))), np.ones((len(d_vals), len(ERROR_P))), []

    for index, d_ in enumerate(d_vals):
        social_cost_, social_cost_l1_, br_iters_ = np.ones((NUM_SIMS, len(ERROR_P))), np.ones((NUM_SIMS, len(ERROR_P))), []
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
        
        for sim, future in enumerate(futures):
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_[sim, :] = sc
            social_cost_l1_[sim, :] = sc_l1

        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_, axis=0), np.sqrt(np.var(social_cost_, axis=0))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_, axis=0), np.sqrt(np.var(social_cost_l1_, axis=0))
        
        avg_sc_str = [str(avg_sc[k]) for k in range(len(ERROR_P))]
        var_sc_str = [str(var_sc[k]) for k in range(len(ERROR_P))]
        avg_sc_l1_str = [str(avg_sc_l1[k]) for k in range(len(ERROR_P))]
        var_sc_l1_str = [str(var_sc_l1[k]) for k in range(len(ERROR_P))]
        row = [str(d_)] + [str(avg_br)] + [str(var_br)] + avg_sc_str + var_sc_str + avg_sc_l1_str + var_sc_l1_str
        csv_writer.writerow(row)
        csv_file.flush()

        br_iters.append(avg_br)
        social_cost[index, :] = avg_sc
        social_cost_l1[index, :] = avg_sc_l1
        equi_found.append( social_cost_.shape[0] / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var[index, :] = var_sc
        social_cost_l1_var[index, :] = var_sc_l1

def sweep_alpha(a_vals, with_l1=False):
    """ Sweep the dimension of the X, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""
    
    csv_file = open("sweep_alpha.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = np.ones((len(a_vals), len(ERROR_P))), np.ones((len(a_vals), len(ERROR_P))), [], []
    social_cost_var, social_cost_l1_var, br_iters_var = np.ones((len(a_vals), len(ERROR_P))), np.ones((len(a_vals), len(ERROR_P))), []
    
    for index, a_ in enumerate(a_vals):
        social_cost_, social_cost_l1_, br_iters_ = np.ones((NUM_SIMS, len(ERROR_P))), np.ones((NUM_SIMS, len(ERROR_P))), []
        print("a = ", a_)
        
        inputs = []
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=N)
            tup = (X, true_Y, N, with_l1, P, a_)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for sim, future in enumerate(futures):
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_[sim, :] = sc
            social_cost_l1_[sim, :] = sc_l1

        br_iters_, social_cost_, social_cost_l1_ = np.array(br_iters_), np.array(social_cost_), np.array(social_cost_l1_)
        avg_br, var_br = np.mean(br_iters_), np.sqrt(np.var(br_iters_))
        avg_sc, var_sc = np.mean(social_cost_, axis=0), np.sqrt(np.var(social_cost_, axis=0))
        avg_sc_l1, var_sc_l1 = np.mean(social_cost_l1_, axis=0), np.sqrt(np.var(social_cost_l1_, axis=0))
        
        avg_sc_str = [str(avg_sc[k]) for k in range(len(ERROR_P))]
        var_sc_str = [str(var_sc[k]) for k in range(len(ERROR_P))]
        avg_sc_l1_str = [str(avg_sc_l1[k]) for k in range(len(ERROR_P))]
        var_sc_l1_str = [str(var_sc_l1[k]) for k in range(len(ERROR_P))]
        row = [str(a_)] + [str(avg_br)] + [str(var_br)] + avg_sc_str + var_sc_str + avg_sc_l1_str + var_sc_l1_str
        csv_writer.writerow(row)
        csv_file.flush()
        
        br_iters.append(avg_br)
        social_cost[index, :] = avg_sc
        social_cost_l1[index, :] = avg_sc_l1
        equi_found.append( social_cost_.shape[0] / NUM_SIMS )

        br_iters_var.append(var_br)
        social_cost_var[index, :] = var_sc
        social_cost_l1_var[index, :] = var_sc_l1

def benchmark_n():
    print("Benchmarking n")
    start = time.time()
    sweep_n([10, 100, 300], with_l1=True)
    end = time.time()
    print(end-start)

def benchmark_d():
    print("Benchmarking d")
    start = time.time()
    sweep_d([1,3,5,7], with_l1=True)
    end = time.time()
    print(end-start)

def benchmark_p():
    print("Benchmarking p")
    start = time.time()
    sweep_p([1.2,3], with_l1=True)
    end = time.time()
    print(end-start)

def benchmark_alpha():
    print("Benchmarking alpha")
    start = time.time()
    sweep_alpha([0.5, 1.0], with_l1=True)
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    #n_vals = np.concatenate([np.arange(10,100,10), np.arange(100,1000,100), np.arange(1000, 4000, 1000)])
    #sweep_n(n_vals, with_l1=True)

    #d_vals = np.concatenate([np.arange(1, 20, 2), np.arange(25, 35, 5), np.arange(30,100,10)])
    #sweep_d(d_vals, with_l1=True)
    
    #p_vals = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #sweep_p(p_vals, with_l1=True)

    a_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_alpha(a_vals, with_l1=True)
    
    #benchmark_n()

