#!/usr/bin/python3

import numpy as np
from p_regression import loss_fn_p, fit_model, get_ith_projection, get_H_matrix

eps = 0.0001

def lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, p):
    """ Compute the best response for agent i, using a smart form of binary searching """
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
    
    
    curr_report = curr_Y[i]
    curr_val = get_ith_projection(X, curr_beta, i)
    true_val = true_Y[i]

    if abs(curr_val - true_val) < eps:
        return curr_report
        
    report = curr_Y.copy() 
    if curr_val > true_val:     # pull down
        #print("Pull Down")
        report[i] = 0
        beta_zero = fit_model(X, report, p=p)
        val = get_ith_projection(X, beta_zero, i)
        if val > true_val:
            return 0
        else:
            return binary_search(0, curr_report, true_val)
    
    elif curr_val < true_val:     # pull up
        #print("Pull Up")
        report[i] = 1
        beta_one = fit_model(X, report, p=p)
        val = get_ith_projection(X, beta_one, i)
        if val < true_val:
            return 1
        else:
            return binary_search(curr_report, 1, true_val)
    else:
        assert False, "should not be here"

def l2_agent_best_response(H, curr_Y, true_Y, i):
    curr_weight = H[i,i]
    n = H.shape[0]
    total = 0

    ith_row = H[i,:].copy()
    ith_row[i] = 0
    br = np.clip((true_Y[i] - np.dot(ith_row, curr_Y)) / curr_weight, 0, 1)
    return br

def test_best_response():
    #### Test 1 - with p=2 and d=1 ####
    X = np.array([[0,1], [0.25, 1], [0.75, 1], [1,1]])
    true_Y = np.array([0, 0.4, 0.6, 1])
    curr_Y = true_Y.copy()
    curr_beta = fit_model(X, curr_Y, p=2)
    H_mat = get_H_matrix(X)
    
    # testing best response
    for i in range(4):
        br_true = l2_agent_best_response(H_mat, curr_Y, true_Y, i)
        br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, 2)
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
        br_true = l2_agent_best_response(H_mat, curr_Y, true_Y, i)
        br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, 2)
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
        br_true = l2_agent_best_response(H_mat, curr_Y, true_Y, i)
        br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, i, 2)
        assert abs(br_approx - br_true) < eps, "TEST: agent best response 3 - FAILED"
    print("TEST: agent best response 3 - PASSED")

    #### Test 3 - with p=4 and d=3 ####
    X, true_Y = generate_data(d=3)
    curr_Y = true_Y.copy()
    curr_beta = fit_model(X, true_Y, p=4)
    br_approx = lp_agent_best_response(X, curr_Y, true_Y, curr_beta, 1, 4)
    
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




