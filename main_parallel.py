import numpy as np
from best_response_algo import best_response_algo
from p_regression import generate_data, fit_model
import matplotlib.pyplot as plt
import csv

import time
import concurrent.futures 

## default values ##
P = 2
N = 100
NUM_SIMS = 1000
ALPHA = 1
XDIM = 1

def plot(x_vals, xlabel, name, br_time_vals, sc_vals, sc_l1_vals=None, log=False):
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
    #plt.savefig(name)
    plt.show()
    
def run_simulation(tup):
    X, true_Y, n_, with_l1, p = tup
    proj_Y = np.matmul(X, fit_model(X, true_Y))
    MSE_h = np.sum(np.power(proj_Y - true_Y, 2))

    if with_l1:
        proj_Y_l1 = np.matmul(X, fit_model(X, true_Y, p=1))
        MSE_l1 = np.sum(np.power(proj_Y_l1 - true_Y, 2))
    else:
        MSE_l1 = 1

    equi_exists, counter, equi_report, beta_equi = best_response_algo(X, true_Y, p)
    social_cost, social_cost_l1, br_iters = 1, 1, 1
    if equi_exists:
        proj = np.matmul(X, beta_equi)
        MSE_eq = np.sum(np.power(proj - true_Y, 2))
     
        if MSE_eq != 0 and MSE_h != 0:
            social_cost = MSE_eq/MSE_h
            social_cost_l1 = MSE_l1/MSE_h
        br_iters = counter*n_
    else:
        print("NO EQUILIBRIUM")
        #fail_equilibrium_writer.writerow([str(X), str(true_Y)])
    return (br_iters, social_cost, social_cost_l1)

def sweep_n(n_vals, with_l1=False, to_plot=True):
    csv_file = open("sweep_n.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    fail_equilibrium = open("fail_equilibrium.csv", mode='w')
    fail_equilibrium_writer = csv.writer(fail_equilibrium, delimiter=',')

    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []

    for n_ in n_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        n_ = int(n_)
        print("n = ", n_)

        inputs = []
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=n_)
            tup = (X, true_Y, n_, with_l1, P)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(12)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for future in futures:
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_.append(sc)
            social_cost_l1_.append(sc_l1)

        avg_br = sum(br_iters_)/len(br_iters_)
        avg_sc = sum(social_cost_)/len(social_cost_)
        avg_sc_l1 = sum(social_cost_l1_)/len(social_cost_l1_)
        
        csv_writer.writerow([str(n_), str(avg_br), str(avg_sc), str(avg_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

    if to_plot:
        if with_l1 == True:
            plot(n_vals, "# of agents", "sweep_n", br_iters, social_cost, social_cost_l1)
        else:
            plot(n_vals, "# of agents", "sweep_n", br_iters, social_cost)

def sweep_p(p_vals, with_l1=False, to_plot=True):
    csv_file = open("sweep_p.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    fail_equilibrium = open("fail_equilibrium.csv", mode='a')
    fail_equilibrium_writer = csv.writer(fail_equilibrium, delimiter=',')

    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []

    for p_ in p_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        print("p = ", p_)
        inputs = []
        
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=N)
            tup = (X, true_Y, N, with_l1, p_)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(12)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for future in futures:
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_.append(sc)
            social_cost_l1_.append(sc_l1)
        
        avg_br = sum(br_iters_)/len(br_iters_)
        avg_sc = sum(social_cost_)/len(social_cost_)
        avg_sc_l1 = sum(social_cost_l1_)/len(social_cost_l1_)
        
        csv_writer.writerow([str(p_), str(avg_br), str(avg_sc), str(avg_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

    if to_plot:
        if with_l1 == True:
            plot(p_vals, "p value", "sweep_p", br_iters, social_cost, social_cost_l1)
        else:
            plot(p_vals, "p value", "sweep_p", br_iters, social_cost)

def sweep_d(d_vals, with_l1=False, to_plot=True):
    csv_file = open("sweep_d.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    fail_equilibrium = open("fail_equilibrium.csv", mode='a')
    fail_equilibrium_writer = csv.writer(fail_equilibrium, delimiter=',')

    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []

    for d_ in d_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        print("d = ", d_)
        inputs = []
        
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=d_, n=N)
            tup = (X, true_Y, N, with_l1, P)
            inputs.append(tup)
            
        executor = concurrent.futures.ProcessPoolExecutor(12)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)

        for future in futures:
            assert(future.done())
            tup = future.result()
            br_time, sc, sc_l1 = tup
            br_iters_.append(br_time)
            social_cost_.append(sc)
            social_cost_l1_.append(sc_l1)

        avg_br = sum(br_iters_)/len(br_iters_)
        avg_sc = sum(social_cost_)/len(social_cost_)
        avg_sc_l1 = sum(social_cost_l1_)/len(social_cost_l1_)
        
        csv_writer.writerow([str(d_), str(avg_br), str(avg_sc), str(avg_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )
    
    if to_plot:
        if with_l1 == True:
            plot(d_vals, "X dimension", "sweep_d", br_iters, social_cost, social_cost_l1)
        else:
            plot(d_vals, "X dimension", "sweep_d", br_iters, social_cost)

def benchmark_n():
    start = time.time()
    sweep_n([10,20,30,40], with_l1=True, to_plot=False)
    end = time.time()
    print(end-start)

def benchmark_d():
    print("Benchmarking d")
    start = time.time()
    sweep_d([1,3,5,7], with_l1=True, to_plot=False)
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    #n_vals = np.concatenate([np.arange(10,100,10), np.arange(100,1050,50), np.arange(1000, 11000, 1000)])
    #sweep_n(n_vals, with_l1=True)

    #d_vals = np.arange(1, 11, 1)
    #sweep_d(d_vals, with_l1=True)
    
    #p_vals = [1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10, 100]
    #sweep_p(p_vals, with_l1=True)
    
    benchmark_d()

