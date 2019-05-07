#!/usr/bin/python3

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import statsmodels.api as sm

def l1_regression_scipy(X, Y):
    def fit(X, beta):
        return np.dot(X, beta)

    def cost_function(beta, X, Y):
        return np.sum(np.abs(Y - fit(X, beta)))
    
    # x0 is a guess for the optimal params. Use the L2 guess
    beta_mse = np.linalg.lstsq(X, Y, rcond=None)[0]
    output = scipy.optimize.minimize(cost_function, beta_mse, args=(X, Y))
    beta_l1 = output.x
    return beta_l1

def l1_regression(X, Y):
    import warnings
    warnings.filterwarnings("ignore")
    results = sm.QuantReg(Y, X).fit(q=0.5)
    return results.params

def loss_fn_p(X, Y, beta, p_):
    return cp.pnorm(cp.matmul(X, beta) - Y, p=p_)**p_

def generate_data(d=1, n=4, sigma=0.5):
    """ d is the dimension of the x point
        n is the number of points
        sigma is the standard deviation
    """
    X = np.random.randn(n, d+1);
    X[:,-1] = np.ones(n)

    mean = np.random.uniform(0,1,1)[0]
    true_Y = np.random.normal(mean, sigma, n)
    true_Y = np.clip(true_Y, 0, 1)
    return X, true_Y

def fit_model(X, Y, p=2, test=False):
    def fit_model_2(X, Y):
        beta_mse = np.linalg.lstsq(X, Y, rcond=None)[0]
        return beta_mse

    if p == 1:
        return l1_regression(X,Y)
    if p == 2 and test == False:
        return fit_model_2(X, Y)
    else:
        loss_fn = loss_fn_p
        d = X.shape[1]
        beta = cp.Variable(d)
        problem = cp.Problem(cp.Minimize(loss_fn(X, Y, beta, p)))
        problem.solve()
        return beta.value

def get_H_matrix(X):
    """ Get the projection matrix for L2 regression """
    H_mat = np.matmul(X, np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), \
            X.transpose()))
    return H_mat

def get_ith_projection(X, beta, i):
    y_bar = np.matmul(X, beta)
    return y_bar[i]

def test():
    """ Use L_2 regression to test out this function, since we know the closed form solution for this """
    eps = 0.0001
    
    #### Test 1 - with arbitrary position and p=2, d=1 ####
    X, true_Y = generate_data()
    beta_mse = fit_model(X, true_Y, p=2, test=False)
    y_bar_mse = np.matmul(X, beta_mse)

    beta_p = fit_model(X, true_Y, p=2, test=True)
    y_bar_p = np.matmul(X, beta_p)

    delta = np.sum(np.power(y_bar_mse - y_bar_p, 2))
    assert delta < eps, "TEST: p regression fitting 1 - FAILED"
    print("TEST: p regression fitting 1 - PASSED")
    
    #### Test 2 - with arbitrary position and p=2, d=3 ####
    X, true_Y = generate_data(d=3)
    beta_mse = fit_model(X, true_Y, p=2, test=False)
    y_bar_mse = np.matmul(X, beta_mse)

    beta_p = fit_model(X, true_Y, p=2, test=True)
    y_bar_p = np.matmul(X, beta_p)

    delta = np.sum(np.power(y_bar_mse - y_bar_p, 2))
    assert delta < eps, "TEST: p regression fitting 2 - FAILED"
    print("TEST: p regression fitting 2 - PASSED")

    #### Test 3 - Co-linear points ####
    X = np.array([[0,1], [0.25, 1], [0.75, 1], [1,1]])
    true_Y = np.array([0, 0.25, 0.75, 1])
    beta_14 = fit_model(X, true_Y, p=1.4)
    y_bar_14 = np.matmul(X, beta_14)

    beta_4 = fit_model(X, true_Y, p=4)
    y_bar_4 = np.matmul(X, beta_4)
    
    delta = np.sum(np.power(true_Y - y_bar_14, 2))
    assert delta < eps, "TEST: p regression fitting 3 - FAILED"
    delta = np.sum(np.power(true_Y - y_bar_4, 2))
    assert delta < eps, "TEST: p regression fitting 3 - FAILED"
    print("TEST: p regression fitting 3 - PASSED")
    
    #### Test 4 - L1 regression ####
    X = np.array([[0,1], [0.25, 1], [0.75, 1], [1,1]])
    true_Y = np.array([0, 1, 0, 1])
     
    beta_1 = fit_model(X, true_Y, p=1)
    y_bar_p = np.matmul(X, beta_1)
    print(y_bar_p)
    true_Y_bar = [0, 0.25, 0.75, 1]
    delta = np.sum(np.power(true_Y_bar - y_bar_p, 2))
    assert delta < eps, "TEST: p regression fitting 4 - FAILED"
    print("TEST: p regression fitting 4 - PASSED")


if __name__ == "__main__":
    test()
