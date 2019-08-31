#!/usr/bin/python3
## Compute the best response under p regression for a single agent

import numpy as np
from p_regression import loss_fn_p, fit_model, get_ith_projection, get_H_matrix
from constants import MIN_Y, MAX_Y
#MIN_Y = 0
#MAX_Y = 1


def to_update(X, curr_Y, true_Y, i, p):
    """ Compute whether an agent needs to update their value or not. Helps with early termination
        Very similar to verify_equilibrium
        TODO: Make verify equilibrium leverage this"""
    
    eps = 0.005*MAX_Y
    proj = np.matmul(X, fit_model(X, curr_Y, p=p))
    n = X.shape[0]

    def in_range(x, y, eps=eps):
        """ Check if x - eps < y < x+eps """
        if x-eps <= y <= x+eps:
            return True
        return False
        
    if in_range(curr_Y[i], MIN_Y) and (proj[i]-eps > true_Y[i] or proj[i] + eps > true_Y[i]):
        return False
    elif in_range(curr_Y[i], MAX_Y) and (proj[i]-eps < true_Y[i] or proj[i] + eps < true_Y[i]):
        return False
    elif not in_range(curr_Y[i], MAX_Y) and not in_range(curr_Y[i], MIN_Y) and in_range(proj[i], true_Y[i]):
        return False
    else:
        return True

def lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, p):
    """ Compute the best response for agent i, using a form of binary searching
        0 means no update from the agent
        1 means an update from the agent
        TODO: clean up some of the cases
    """
    eps = 0.0001*MAX_Y
    def binary_search(min_val, max_val, true_val):
        mid = min_val + ((max_val - min_val) / 2)
        mid_report = curr_Y.copy()
        mid_report[i] = mid
        beta_mid = fit_model(X, mid_report, p=p)
        mid_val = get_ith_projection(X, beta_mid, i)

        if abs(mid_val - true_val) < eps:
            return mid
        elif mid_val > true_val:    # Need to go lower
            return binary_search(min_val, mid, true_val)
        else:                       # Need to go lower
            return binary_search(mid, max_val, true_val)
    
    if to_update(X, curr_Y, true_Y, i, p) == False:
        return 0, curr_Y[i]

    curr_report = curr_Y[i]
    old_report = curr_report
    curr_val = get_ith_projection(X, curr_beta, i)
    true_val = true_Y[i]

    # possibly redundant
    if abs(curr_val - true_val) < eps:
        return 0, curr_report
        
    report = curr_Y.copy() 
    if curr_val > true_val:     # pull down the hyperplane
        report[i] = MIN_Y
        beta_zero = fit_model(X, report, p=p)
        val = get_ith_projection(X, beta_zero, i)
        if val > true_val:
            if old_report == MIN_Y:
                return 0, MIN_Y
            else:
                return 1, MIN_Y
        else:
            ret = binary_search(MIN_Y, curr_report, true_val)
            if old_report == ret:
                return 0, ret
            else:
                return 1, ret
            
    elif curr_val < true_val:     # pull up the hyperplane
        #print("Pull Up")
        report[i] = MAX_Y
        beta_one = fit_model(X, report, p=p)
        val = get_ith_projection(X, beta_one, i)
        if val < true_val:
            if old_report == MAX_Y:
                return 0, MAX_Y
            else:
                return 1, MAX_Y
        else:
            ret = binary_search(curr_report, MAX_Y, true_val)
            if old_report == ret:
                return 0, ret
            else:
                return 1, ret
    else:
        assert False, "should not be here"

def l2_agent_best_response(H, curr_Y, true_Y, i):
    """ The best response for an agent under 2-norm.
        Note bar{y} = H*tilde(y), where H is the projection matrix
        Pretty simple, but a bit obsfucated as I wrote it in vectorized
        form for efficient
    """
    
    eps = 0.0001*MAX_Y
    curr_weight = H[i,i]
    curr_val = curr_Y[i]
    n = H.shape[0]
    total = 0

    ith_row = H[i,:].copy()
    ith_row[i] = 0
    br = np.clip((true_Y[i] - np.dot(ith_row, curr_Y)) / curr_weight, MIN_Y, MAX_Y)
   
    if (br - curr_val) < eps:
        return 0, br
    else:
        return 1, br

def test_best_response():
    eps = 0.005*MAX_Y
    #### Test 1 - with p=2 and d=1 ####
    X = np.array([[0,1], [0.25, 1], [0.75, 1], [1,1]])
    true_Y = np.array([0, 0.4, 0.6, 1])
    curr_Y = true_Y.copy()
    curr_beta = fit_model(X, curr_Y, p=2)
    H_mat = get_H_matrix(X)
    
    # testing best response
    for i in range(4):
        _, br_true = l2_agent_best_response(H_mat, curr_Y, true_Y, i)
        _, br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, 2)
        assert abs(br_approx - br_true) < eps, "TEST: agent best response 1 - FAILED"
    print("TEST: agent best response 1 - PASSED")

    #### Test 2 - with p=2 and d=1 #####
    X = np.array([[2.109, 1], [-2.6212, 1], [2.903, 1], [-0.634, 1]])
    true_Y = np.array([0, 0.8596, 0.633, 0.749])
    curr_Y = true_Y.copy()
    curr_beta = fit_model(X, curr_Y, p=2)
    H_mat = get_H_matrix(X)
    
    # testing best response
    for i in range(4):
        _, br_true = l2_agent_best_response(H_mat, curr_Y, true_Y, i)
        _, br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, 2)
        assert abs(br_approx - br_true) < eps, "TEST: agent best response 2 - FAILED"
    print("TEST: agent best response 2 - PASSED")

    #### Test 3 - with p=2 and d=3 ####
    from p_regression import generate_data
    X, true_Y = generate_data(d=3)
    curr_Y = true_Y.copy()
    curr_beta = fit_model(X, true_Y, p=2)
    H_mat = get_H_matrix(X)
    
    # testing best response
    for i in range(4):
        _, br_true = l2_agent_best_response(H_mat, curr_Y, true_Y, i)
        _, br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, 2)
        assert abs(br_approx - br_true) < eps, "TEST: agent best response 3 - FAILED"
    print("TEST: agent best response 3 - PASSED")

    #### Test 3 - with p=4 and d=3 ####
    X, true_Y = generate_data(d=3)
    curr_Y = true_Y.copy()
    curr_beta = fit_model(X, true_Y, p=4)
    _, br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, 1, 4)
    
    curr_Y[1] = br_approx
    br_utl = abs(true_Y[1] - get_ith_projection(X, fit_model(X, curr_Y, p=4), 1))
    
    test_Y = true_Y.copy()
    for j in np.linspace(0,1,0.01):
        test_Y[1] = j
        test_utl = abs(true_Y[1] - get_ith_projection(X, fit_model(X, test_Y, p=4), 1))
        assert test_utl > br_utl, "TEST: agent best response 3 - FAILED"
    print("TEST: agent best response 3 - PASSED")

if __name__ == "__main__":
    test_best_response()




