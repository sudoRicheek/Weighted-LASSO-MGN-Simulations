"""
Adithya, Richeek, Aaron, 2022.
Provides helper functions to perform LASSO and Weighted LASSO.
While some of this functionality exists in Ritesh's code, having an easy one-stop plug-and-play
outlet for the optimization that we are familiar with will be a great help.
"""

import re
from globals import *
from vars import rescale_and_center

import cvxpy as cp
import numpy as np
from tqdm import tqdm
import math

def mse_error(x, y, A):
    """
    Simple mean-squared error, mainly for visualization purposes.
    Parameters:
    @x - The cvxpy variable representing x
    @Y - n x 1 result vector
    @A - n x p sensing matrix
    """
    l2_term = cp.norm2(y - np.matmul(A,x))**2
    mse = l2_term.value / x.shape[0]
    return mse

def get_weighted_lasso_objective(x, y, A, D, lambd=1):
    """
    The objective in weighted LASSO is
    || y - Ax ||^2 + lambd \sum_{i=1}^p |d_i x_i|
    Parameters:
    @x - The cvxpy variable representing x
    @Y - n x 1 result vector
    @A - n x p sensing matrix
    @d - p x 1 matrix to be multiplied pre-L1-norm.
    @lambd - scalar regularization parameter
    """
    l2_term = cp.norm2(y - A@x)**2
    regularizer = lambd*cp.norm1(D@x)
    return (l2_term + regularizer)
    
def get_lasso_objective(x, y, A, lambd=1):
    """
    The weighted lasso objective, with all weights equalling 1.
    """
    D = np.eye(A.shape[-1])
    return get_weighted_lasso_objective(x, y, A, D, lambd)

def solve_weighted_lasso(y, A, D=None, lambd=1):
    """
    Performs the weighted lasso optimization. Returns the optimal x found.
    """
    x = cp.Variable(A.shape[-1])
    if D is None:
        objective = get_lasso_objective(x, y, A, lambd)
    else:
        objective = get_weighted_lasso_objective(x, y, A, D, lambd)
    problem = cp.Problem(cp.Minimize(objective))    
    problem.solve(verbose=False)
    return x.value

def best_weighted_lasso_with_fudge_factor(x_real, y, A, D=None):
    """
    Search for best lambda.
    """
    candidates = [1e-3, 1e-2, 3e-2, 1e-1, 5e-1, 1, 10, 30, 100, 1000]
    best_x = None
    best_dist = 1e18
    best_lambd = None
    for lambd in candidates:
        try:
            this_x = solve_weighted_lasso(y, A, D, lambd)
        except:
            continue
        this_dist = np.linalg.norm(this_x - x_real)
        if this_dist < best_dist:
            best_dist = this_dist
            best_x = this_x
            best_lambd = lambd
    return best_x, best_lambd

def monte_carlo_simulation(generator, num=500, debug=False):
    msum = 0
    msqsum = 0
    rrange = range(num)
    if debug:
        rrange = tqdm(rrange)
    for _ in rrange:
        x_real, y, A = generator.next()
        Ahat, yhat = rescale_and_center(A, y)
        Dk = get_d_vec(A, y, generator.n, generator.p, generator.q, generator.sigma)
        Dk *= math.sqrt(Dk.shape[0]) / np.linalg.norm(Dk)
        Dk = np.diag(Dk)
        # d = get_d(A, y, generator.n, generator.p, generator.q, generator.sigma)
        # x = solve_weighted_lasso(yhat, Ahat, D=Dk, lambd=1)
        # x = solve_weighted_lasso(yhat, Ahat, D=None, lambd=d)
        best_x, best_lambd = best_weighted_lasso_with_fudge_factor(x_real, y, A, Dk)
        rrmse = np.sqrt(np.linalg.norm(best_x-x_real)**2)/np.linalg.norm(x_real)
        msum += rrmse
        msqsum += rrmse**2
    mean = msum / num
    var = (msqsum/num) - mean**2
    return mean, math.sqrt(var)

def get_V(A, q):
    A = A.T
    p, n = A.shape[0], A.shape[1]
    V = (n*A - np.sum(A, axis=-1, keepdims=True))/(n*(n-1)*q*(1-q))
    return np.square(V)

def get_W(A, q):
    # TODO check V v/s V.T for shape
    V = get_V(A,q)
    W = np.sum(np.matmul(V, A), axis=-1)
    return np.max(W)

def gfunc(theta, n):
    g = (1 - math.exp(-theta))**(1/n)
    g = -math.log(1-g)
    return g

def get_Nhat(y, n, p, q, sigma, return_term=False):
    denom = n*q - math.sqrt(6*n*q*(1-q)*math.log(p)) - max(q,1-q)*math.log(p)
    kappa = sigma*math.log(1+qa)
    term = kappa*math.sqrt(2*gfunc(3*math.log(p), n))
    assert(term < 1)
    term = kappa*math.sqrt(6*math.log(p))/(1-term)
    numer = np.sum(y) + np.sqrt(np.sum(np.square(y)))*term
    ans = numer/denom
    if return_term:
        return ans, term
    else:
        return ans

def get_d(A, y, n, p, q, sigma, c=126):
    Nhat = get_Nhat(y,n,p,q,sigma)
    W = get_W(A,q)
    kappa = sigma*math.log(1+qa)
    term1 = kappa*Nhat*math.sqrt(6*W*math.log(p))
    term2 = c*((3*math.log(p)/n) + (9*max(q**2, (1-q)**2)*(math.log(p)**2)/(q*(1-q)*n**2)))*Nhat
    d = term1+term2
    return d

def get_d_vec(A, y, n, p, q, sigma, c=126):
    Nhat, term = get_Nhat(y,n,p,q,sigma,return_term=True)
    V = get_V(A,q)
    y2 = np.square(y)
    term1 = np.sqrt(np.matmul(V, y2))*term
    term2 = c*((3*math.log(p)/n) + (9*max(q**2, (1-q)**2)*(math.log(p)**2)/(q*(1-q)*n**2)))*Nhat
    ret = term1 + term2
    return ret    

def _test_lasso(n=15, p=40, sparsity=0.3, noise_constant=0):
    """
    A tester function to ensure the LASSO optimization works.
    """
    A = np.random.randn(n, p)
    x_real = np.random.randn(p)
    zero_indices = np.random.permutation(p)[int(p*sparsity):]
    for ind in zero_indices:
        x_real[ind] = 0
    y = np.matmul(A, x_real) + noise_constant*np.random.randn(A.shape[-2])
    
    x_rec = solve_weighted_lasso(y, A)
    
    error = (np.linalg.norm(x_rec - x_real)**2) / x_real.shape[0]
    print("Error: ", error)

if __name__ == '__main__':
    _test_lasso()