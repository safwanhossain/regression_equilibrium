#!/usr/bin/python3
# Generate data for simulations and code to fit the regression lines for a given p

import cvxpy as cp
import numpy as np
import scipy.optimize
import statsmodels.api as sm

def l1_regression_scipy(X, Y):
    """ L1 regression using scipy - unstable"""
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
    """ L1 regression using stats packagage - stable and use this"""
    import warnings
    warnings.filterwarnings("ignore")
    results = sm.QuantReg(Y, X).fit(q=0.5)
    return results.params

def loss_fn_p(X, Y, beta, p_):
    """ Lp loss function - used for cxvpy to compute the optimal regression params"""
    return cp.pnorm(cp.matmul(X, beta) - Y, p=p_)**p_

def generate_data_naive(d=1, n=4, sigma=0.5):
    """ d is the dimension of the x point
        n is the number of points
        sigma is the standard deviation
        
        Need to ensure all points have y vals in [0,1], but don't wanna clip too much
        SUFFERS FROM ~40% CLIPPING
        VERY BAD AND DON'T USE
    """
    X = np.random.randn(n, d+1);
    X[:,-1] = np.ones(n)

    mean = np.random.uniform(0,1,1)[0]
    true_Y = np.random.normal(mean, sigma, n)
    before = true_Y
    true_Y = np.clip(true_Y, 0, 1)
    return X, true_Y
                
def generate_data(d=1, n=4, sigma=0.5):
    """ d is the dimension of the x point
        n is the number of points
        sigma is the standard deviation

        Generate the X values according to Gaussian distribution centered at 0
        Generate the beta values according to uniform distribution [0,1]
        Get the corresponding projections and then add noise
        Then normalize (not clip!!) to ensure values are between 0 and 1
        USE THIS - VERY GOOD
    """
    X = sigma*np.random.randn(n, d);
    X = np.concatenate([X, np.ones((n,1))], axis=1)
    
    beta_vec = np.concatenate([np.random.uniform(-1,1,d), np.random.uniform(-1,1,1)])
    noise_vec = (sigma/2)*np.random.randn(n)
    true_Y = np.matmul(X, beta_vec) + noise_vec

    # normalize
    true_Y = true_Y + abs(min(0, min(true_Y)))
    true_Y = true_Y / max(1.0, max(true_Y))

    # safety (but never really needed)
    clip_Y = np.clip(true_Y, 0, 1) 
    return X, clip_Y

def fit_model(X, Y, p=2, test=False):
    """ Find the optimal regression paramter for the input and a choice of p
    for p != 2, use convex optimization lib - otherwise, use standard least squares"""
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
    if X.shape[0] == X.shape[1] - 1:
        return np.eye(X.shape[0])
    else:
        H_mat = np.matmul(X, np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), \
                X.transpose()))
        return H_mat

def get_ith_projection(X, beta, i):
    """ Get the ith element of the projection vector"""
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
    true_Y_bar = [0, 0.25, 0.75, 1]
    delta = np.sum(np.power(true_Y_bar - y_bar_p, 2))
    assert delta < eps, "TEST: p regression fitting 4 - FAILED"
    print("TEST: p regression fitting 4 - PASSED")

    #### Test 5 - L2 regression where d=n ####
    X, Y = generate_data(d=5, n=5, sigma=0.5)
    beta = fit_model(X, Y, p=1.2)
    y_fit = np.matmul(X, beta)
    delta = np.sum(np.power(y_fit - Y, 2))
    assert delta < eps, "TEST: p regression fitting 5 - FAILED"
    print("TEST: p regression fitting 5 - PASSED")
    
if __name__ == "__main__":
    test()






