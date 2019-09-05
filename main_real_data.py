#!/usr/bin/python3

import numpy as np
from best_response_algo import best_response_algo
from p_regression import generate_data, fit_model, generate_regression_matrix_taiwan, generate_regression_matrix_california
import sys, csv
from constants import P, N, NUM_SIMS, ALPHA, XDIM, NUM_CORES, ERROR_P, MAX_Y

import time
import concurrent.futures 

def run_simulation(tup):
    "Find the nash equilbrium for the given input and get the necessary stats"""
    a = None
    if len(tup) == 5:
        X, true_Y, n_, with_l1, p = tup
    if len(tup) == 6:
        X, true_Y, n_, with_l1, p, a = tup

    proj_Y = [np.matmul(X, fit_model(X, true_Y,p=k)) for k in ERROR_P]
    errors_honest = [np.sum(np.power(np.abs(proj_Y[j] - true_Y), k)) for j, k in enumerate(ERROR_P)]

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

        if errors_equi[0] != 0 and errors_equi[0] != 0:
            social_cost = [errors_equi[k]/errors_honest[k] for k in range(len(ERROR_P))]
            social_cost_l1 = [errors_l1[k]/errors_honest[k] for k in range(len(ERROR_P))]
        br_iters = iters
    else:
        print("NO EQUILIBRIUM")
    return (br_iters, social_cost, social_cost_l1)

def sweep_alpha(a_vals, data, with_l1=False, to_plot=True):
    """ Sweep the dimension of the X, and run NUM_SIMS number fo simulations for each values
    Parallelize the simulations across multiple processes
    compute the best response iterations, social cost of equi and social cost of L1"""
    
    dataset, num_entries, dimension = data
    print("Running:", dataset)

    csv_file = open("sweep_alpha.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    social_cost, social_cost_l1, br_iters, equi_found = np.ones((len(a_vals), len(ERROR_P))), np.ones((len(a_vals), len(ERROR_P))), [], []
    social_cost_var, social_cost_l1_var, br_iters_var = np.ones((len(a_vals), len(ERROR_P))), np.ones((len(a_vals), len(ERROR_P))), []

    if dataset == "taiwan_real_estate":
        X, true_Y = generate_regression_matrix_taiwan(num_entries, dimension)
    elif dataset == "california_real_estate":
        X, true_Y = generate_regression_matrix_california(num_entries, dimension)

    for index, a_ in enumerate(a_vals):
        social_cost_, social_cost_l1_, br_iters_ = np.ones((NUM_SIMS, len(ERROR_P))), np.ones((NUM_SIMS, len(ERROR_P))), []
        print("a = ", a_)
        
        inputs = []
        for i in range(NUM_SIMS):
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
            X, true_Y = generate_regression_matrix("taiwan_real_estate.csv", 414, 6)
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
        csv_file.flush()

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
       
if __name__ == "__main__":
    """ Run this file as ./main_real_data <california/taiwan> """
    
    a_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    data_taiwan = ("taiwan_real_estate", 414, 6)
    data_california = ("california_real_estate", 2000, 8)
    assert ERROR_P == [2], "See constants.py"

    dataset = sys.argv[1].lower()
    if dataset == "california":
        assert MAX_Y == 2000, "See constants.py"
        sweep_alpha(a_vals, data_california, with_l1=True, to_plot=False)

    elif dataset == "taiwan":
        assert MAX_Y == 100, "See constants.py"
        sweep_alpha(a_vals, data_taiwan, with_l1=True, to_plot=False)

    else:
        print("Invalid Data Set")

