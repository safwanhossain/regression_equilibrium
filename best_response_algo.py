#!/usr/bin/python3
## Compute the best response algorithm that obtains a Nash Equilibrium

import numpy as np
from p_best_response import lp_agent_best_response, l2_agent_best_response
from p_regression import generate_data, loss_fn_p, fit_model, get_ith_projection, get_H_matrix

MAX_COUNT = 2000

""" Different eps values are needed in different functions primarily due to the sensitivity
 of the algorithm to certain pathological inputs and high number of agents"""

def best_response_algo(X, true_Y, p, test=False, alpha=None):
    """ If p == 2, we will be using the analytic best response algorithm
        Otherwise, we will be going for an approximate best response 
        
        Each round, all agents are given a chance to update their values according
        to their best response. When a round arises, wherein the agents don't update
        (within some eps) then we have reached an approximate equilibrium.
        """
    n = X.shape[0] 
    eps = 0.0001
    
    agents = np.array([i for i in range(n)])
    if alpha != None:
        agents = np.random.choice(agents, int(len(agents)*alpha), replace=False)
    
    yLast = np.ones(n)
    yNew = true_Y.copy()
    if p == 2:
        H_mat = get_H_matrix(X)

    counter = 0
    br_iters = 0
    curr_beta = [-1]
    while np.linalg.norm(yLast - yNew) > eps and counter < MAX_COUNT:
        yLast = yNew.copy()
        curr_beta = fit_model(X, yNew, p=p)
        
        for j in agents:
            if p == 2 and test == False:
                update, br = l2_agent_best_response(H_mat, yNew, true_Y, j)
            else:
                update, br = lp_agent_best_response(X, yNew, true_Y, curr_beta, j, p)        
            yNew[j] = br
            curr_beta = fit_model(X, yNew, p=p)
            br_iters += update
        counter += 1
        
    equi_report = yNew
    if curr_beta[0] == -1:
        curr_beta = fit_model(X, equi_report, p=p)
    
    if np.linalg.norm(yLast - equi_report) <= eps and verify_equilibrium(X, equi_report, true_Y, p, agents):
        return (True, counter, br_iters, equi_report, curr_beta)
    else:
        return (False, counter, br_iters, equi_report, curr_beta)

def verify_equilibrium(X, equi_Y, true_Y, p, agents=None):
    """ Verify if this is an equilibrium. The idea (mentioned in the paper) is: at equilbrium, the
    agent either reports 0 and the line passes above his preferred point, reports 1 and it passes
    below his preferred point, or reports something in (0,1) and the line goes right through
    his preferred point"""
    
    eps = 0.005
    proj = np.matmul(X, fit_model(X, equi_Y, p=p))
    n = X.shape[0]
   
    def in_range(x, y, eps=eps):
        """ Check if x - eps < y < x+eps """
        if x-eps <= y <= x+eps:
            return True
        return False
    
    for i in agents:
        if in_range(equi_Y[i], 0) and (proj[i]-eps > true_Y[i] or proj[i] + eps > true_Y[i]):
            continue
        elif in_range(equi_Y[i], 1) and (proj[i]-eps < true_Y[i] or proj[i] + eps < true_Y[i]):
            continue
        elif not in_range(equi_Y[i], 1) and not in_range(equi_Y[i], 0) and in_range(proj[i], true_Y[i]):
            continue
        else:
            return False
    return True

def test_br_equilibrium():
    eps = 0.01
    
    #### Test 1 with d=1 ####
    X = np.array([[0,1], [0.25, 1], [0.75, 1], [1,1]])
    true_Y = np.array([0, 0.4, 0.6, 1])
    equi_Y = np.array([0, 1, 0, 1])
    equi_exists, _, _, equi_report, _ = best_response_algo(X, true_Y, p=2, test=True)
    assert np.linalg.norm(equi_Y - equi_report) < eps, "TEST: best response equilibrium 1 - FAILED"
    print("TEST: best response equilibrium 1 - PASSED") 
    
    #### Test 2 with d=1 ####
    #X = np.array([[2.109, 1], [-2.6212, 1], [2.903, 1], [-0.634, 1]])
    #true_Y = np.array([0, 0.8596, 0.633, 0.749])
    #equi_Y = np.array([0, 0.8344, 1, 0.9221])
    #equi_exists, _, _, equi_report, _ = best_response_algo(X, true_Y, p=2, test=True)
    #print(np.linalg.norm(equi_Y - equi_report))
    #assert np.linalg.norm(equi_Y - equi_report) < eps, "TEST: best response equilibrium 2 - FAILED"
    #print("TEST: best response equilibrium 1 - PASSED") 

    #### Test 3 with d=2 ####
    X = np.array([[0.52, 0.8513, 1], [0.477, 0.5612, 1], [0.4108, 0.1018, 1], [0.3631, 0.0954, 1]])
    true_Y = np.array([0.6989, 0.5163, 0.8786, 0.5372])
    equi_Y = np.array([1, 0, 1, 0.5475])
    equi_exists, _, _, equi_report, _ = best_response_algo(X, true_Y, p=2, test=True)
    assert np.linalg.norm(equi_Y - equi_report) < eps, "TEST: best response equilibrium 3 - FAILED"
    print("TEST: best response equilibrium 3 - PASSED") 

if __name__ == "__main__":
    test_br_equilibrium()









