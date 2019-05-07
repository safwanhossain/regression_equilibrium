import numpy as np
from best_response_algo import best_response_algo
from p_regression import generate_data, fit_model
import matplotlib.pyplot as plt
import csv

## default values ##
P = 2
N = 100
NUM_SIMS = 100
ALPHA = 1
XDIM = 1

def plot(x_vals, xlabel, name, br_time_vals, sc_vals, sc_l1_vals=None):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('BR Iterations', color=color)
    ax1.plot(x_vals, br_time_vals, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Social Cost', color=color)
    ax2.plot(x_vals, sc_vals, color=color)
    if sc_l1_vals is not None:
        ax2.plot(x_vals, sc_l1_vals, color=color, linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

def sweep_n(n_vals, with_l1=False):
    csv_file = open("sweep_n.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    fail_equilibrium = open("fail_equilibrium.csv", mode='w')
    fail_equilibrium_writer = csv.writer(fail_equilibrium, delimiter=',')

    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []

    for n_ in n_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        n_ = int(n_)
        print("n = ", n_)
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=n_)
            proj_Y = np.matmul(X, fit_model(X, true_Y))
            MSE_h = np.sum(np.power(proj_Y - true_Y, 2))

            if with_l1:
                proj_Y_l1 = np.matmul(X, fit_model(X, true_Y, p=1))
                MSE_l1 = np.sum(np.power(proj_Y_l1 - true_Y, 2))
            else:
                MSE_l1 = 1

            equi_exists, counter, equi_report, beta_equi = best_response_algo(X, true_Y, P)
            if equi_exists:
                proj = np.matmul(X, beta_equi)
                MSE_eq = np.sum(np.power(proj - true_Y, 2))
             
                if MSE_eq == 0 and MSE_h == 0:
                    social_cost_.append(1)
                    social_cost_l1_.append(1)
                else:
                    social_cost_.append(MSE_eq/MSE_h)
                    social_cost_l1_.append(MSE_l1/MSE_h)
                br_iters_.append(counter*n_)
            else:
                print("NO EQUILIBRIUM")
                fail_equilibrium_writer.writerow([str(X), str(true_Y)])

        avg_br = sum(br_iters_)/len(br_iters_)
        avg_sc = sum(social_cost_)/len(social_cost_)
        avg_sc_l1 = sum(social_cost_l1_)/len(social_cost_l1_)
        
        csv_writer.writerow([str(n_), str(avg_br), str(avg_sc), str(avg_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

    if with_l1 == True:
        plot(n_vals, "# of agents", "sweep_n", br_iters, social_cost, social_cost_l1)
    else:
        plot(n_vals, "# of agents", "sweep_n", br_iters, social_cost)

def sweep_p(p_vals, with_l1=False):
    csv_file = open("sweep_p.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    fail_equilibrium = open("fail_equilibrium.csv", mode='a')
    fail_equilibrium_writer = csv.writer(fail_equilibrium, delimiter=',')

    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []

    for p_ in p_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        print("p = ", p_)
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=XDIM, n=N)
            proj_Y = np.matmul(X, fit_model(X, true_Y, p=2))
            MSE_h = np.sum(np.power(proj_Y - true_Y, 2))

            if with_l1:
                proj_Y_l1 = np.matmul(X, fit_model(X, true_Y, p=1))
                MSE_l1 = np.sum(np.power(proj_Y_l1 - true_Y, 2))
            else:
                MSE_l1 = 1

            equi_exists, counter, equi_report, beta_equi = best_response_algo(X, true_Y, p_, test=True)
            if equi_exists:
                proj = np.matmul(X, beta_equi)
                MSE_eq = np.sum(np.power(proj - true_Y, 2))
             
                if MSE_eq == 0 and MSE_h == 0:
                    social_cost_.append(1)
                    social_cost_l1_.append(1)
                else:
                    social_cost_.append(MSE_eq/MSE_h)
                    social_cost_l1_.append(MSE_l1/MSE_h)
                br_iters_.append(counter*N)
            else:
                print("NO EQUILIBRIUM")
                fail_equilibrium_writer.writerow([str(X), str(true_Y)])
            print(i)

        avg_br = sum(br_iters_)/len(br_iters_)
        avg_sc = sum(social_cost_)/len(social_cost_)
        avg_sc_l1 = sum(social_cost_l1_)/len(social_cost_l1_)
        
        csv_writer.writerow([str(p_), str(avg_br), str(avg_sc), str(avg_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

    if with_l1 == True:
        plot(p_vals, "p value", "sweep_p", br_iters, social_cost, social_cost_l1)
    else:
        plot(p_vals, "p value", "sweep_p", br_iters, social_cost)

def sweep_d(d_vals, with_l1=False):
    csv_file = open("sweep_d.csv", mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    fail_equilibrium = open("fail_equilibrium.csv", mode='a')
    fail_equilibrium_writer = csv.writer(fail_equilibrium, delimiter=',')

    social_cost, social_cost_l1, br_iters, equi_found = [], [], [], []

    for d_ in d_vals:
        social_cost_, social_cost_l1_, br_iters_ = [], [], []
        print("d = ", d_)
        for i in range(NUM_SIMS):
            X, true_Y = generate_data(d=d_, n=N)
            proj_Y = np.matmul(X, fit_model(X, true_Y, p=2))
            MSE_h = np.sum(np.power(proj_Y - true_Y, 2))

            if with_l1:
                proj_Y_l1 = np.matmul(X, fit_model(X, true_Y, p=1))
                MSE_l1 = np.sum(np.power(proj_Y_l1 - true_Y, 2))
            else:
                MSE_l1 = 1

            equi_exists, counter, equi_report, beta_equi = best_response_algo(X, true_Y, P, test=True)
            if equi_exists:
                proj = np.matmul(X, beta_equi)
                MSE_eq = np.sum(np.power(proj - true_Y, 2))
             
                if MSE_eq == 0 and MSE_h == 0:
                    social_cost_.append(1)
                    social_cost_l1_.append(1)
                else:
                    social_cost_.append(MSE_eq/MSE_h)
                    social_cost_l1_.append(MSE_l1/MSE_h)
                br_iters_.append(counter*N)
            else:
                print("NO EQUILIBRIUM")
                fail_equilibrium_writer.writerow([str(X), str(true_Y)])
            print(i)

        avg_br = sum(br_iters_)/len(br_iters_)
        avg_sc = sum(social_cost_)/len(social_cost_)
        avg_sc_l1 = sum(social_cost_l1_)/len(social_cost_l1_)
        
        csv_writer.writerow([str(d_), str(avg_br), str(avg_sc), str(avg_sc_l1)])

        br_iters.append(avg_br)
        social_cost.append(avg_sc)
        social_cost_l1.append(avg_sc_l1)
        equi_found.append( len(social_cost_) / NUM_SIMS )

    if with_l1 == True:
        plot(d_vals, "X dimension", "sweep_d", br_iters, social_cost, social_cost_l1)
    else:
        plot(d_vals, "X dimension", "sweep_d", br_iters, social_cost)

if __name__ == "__main__":
    n_vals = np.concatenate([np.arange(10,100,10), np.arange(100,1100,100)])
    sweep_n(n_vals, with_l1=True)

    d_vals = np.arange(1, 11, 1)
    sweep_d(d_vals, with_l1=True)
    
    p_vals = [1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10, 100]
    sweep_p(p_vals, with_l1=True)

