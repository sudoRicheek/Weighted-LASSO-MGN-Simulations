"""
Adithya, Richeek, Aaron, 2022.
Provides helper functions to perform LASSO and Weighted LASSO.
While some of this functionality exists in Ritesh's code, having an easy one-stop plug-and-play
outlet for the optimization that we are familiar with will be a great help.
"""

from random import betavariate
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

def solve_weighted_lasso(y, A, D=None, lambd=1, verbose=False):
    """
    Performs the weighted lasso optimization. Returns the optimal x found.
    """
    x = cp.Variable(A.shape[-1])
    if D is None:
        objective = get_lasso_objective(x, y, A, lambd)
    else:
        objective = get_weighted_lasso_objective(x, y, A, D, lambd)
    constraints = [ x >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)    
    problem.solve(verbose=verbose)
    return x.value

def best_weighted_lasso_with_fudge_factor(x_real, y, A, D=None, normalize=False):
    """
    Search for best lambda.
    """
    candidates_log = math.log(10) * np.linspace(-5, 2, 8)
    candidates = np.exp(candidates_log)
    best_x = np.zeros(x_real.shape)
    best_dist = 1e12
    best_lambd = None
    for lambd in candidates:
        try:
            this_x = solve_weighted_lasso(y, A, D, lambd)
            if this_x is None:
                continue
        except:
            continue
        this_dist = np.linalg.norm(this_x - x_real)
        if this_dist < best_dist:
            best_dist = this_dist
            best_x = this_x
            best_lambd = lambd
    return best_x, best_lambd

def monte_carlo_simulation(generator, num=100, num_calibration=5, debug=False, \
    single_weight=False, best_lambd=None):
    mean_rrmse = 0
    mean_sensitivity = 0
    mean_specificity = 0
    mean_mcc = 0
    num_for_mean = 0
    rrange = range(num)
    crange = range(num_calibration)
    if debug:
        rrange = tqdm(rrange)
        crange = tqdm(crange)
    if best_lambd is None:
        candidates_log = math.log(10) * np.linspace(-5, 2, 20)
        candidates = np.exp(candidates_log)
        # Find best fudge factor
        rrmse_calib = 1e8
        for candidate in candidates:
            print("\nCalibration: checking with", candidate,"\n")
            this_rrmse = 0
            for _ in tqdm(crange):
                if this_rrmse > rrmse_calib*num_calibration:
                    break
                x_real = None
                while x_real is None:
                    x_real, y, A = generator.next()
                Ahat, yhat = rescale_and_center(A, y)
                if single_weight:
                    d = get_d(A, y, generator.n, generator.p, generator.q, generator.sigma)
                    Dk = np.eye(x_real.shape[0])*d
                else:
                    Dk = get_d_vec(A, y, generator.n, generator.p, generator.q, generator.sigma)
                    Dk *= math.sqrt(Dk.shape[0]) / np.linalg.norm(Dk)
                    Dk = np.diag(Dk)
                try:
                    best_x = solve_weighted_lasso(yhat, Ahat, Dk, \
                        candidate, verbose=False)
                    if best_x is None:
                        this_rrmse = 1e11
                        break
                except:
                    print("Solver failed for this calibration case - skipping")
                    this_rrmse = 1e11
                    break
                
                this_rrmse += get_rrmse(x_real, best_x)
            this_rrmse /= num_calibration
            if this_rrmse < rrmse_calib:
                best_lambd = candidate
                rrmse_calib = this_rrmse
    extra_info = {'best_lambd' : best_lambd}
    d_avg = 0
    dk_avg = 0
    num_mcc_for_mean = 0
    for _ in rrange:
        x_real = None
        while x_real is None:
            x_real, y, A = generator.next()
        Ahat, yhat = rescale_and_center(A, y)
        if single_weight:
            d = get_d(A, y, generator.n, generator.p, generator.q, generator.sigma)
            Dk = np.eye(x_real.shape[0])*d
            d_avg += d
        else:
            Dk = get_d_vec(A, y, generator.n, generator.p, generator.q, \
                generator.sigma)
            Dk *= math.sqrt(Dk.shape[0]) / np.linalg.norm(Dk)
            dk_avg += Dk
            Dk = np.diag(Dk)
        try:
            best_x = solve_weighted_lasso(yhat, Ahat, Dk, best_lambd, \
                verbose=False)
            if best_x is None:
                continue
            num_for_mean += 1
        except:
            continue
        
        mean_rrmse += get_rrmse(x_real, best_x)
        sensitivity, specificity, mcc = sensitivity_specificity_and_mcc(\
            x_real, best_x)
        mean_sensitivity += sensitivity
        mean_specificity += specificity
        if mcc is not None:
            num_mcc_for_mean += 1
            mean_mcc += mcc
    
    if not single_weight:
        dk_avg = (dk_avg/num).tolist()
    extra_info['d_avg'] = d_avg / num 
    extra_info['dk_avg'] = dk_avg   
    mean_rrmse /= num_for_mean
    mean_sensitivity /= num_for_mean
    mean_specificity /= num_for_mean
    mean_mcc /= num_mcc_for_mean
    return mean_rrmse, mean_sensitivity, mean_specificity, mean_mcc, \
        extra_info

def get_V(A, q):
    A = A.T
    p, n = A.shape[0], A.shape[1]
    V = (n*A - np.sum(A, axis=-1, keepdims=True))/(n*(n-1)*q*(1-q))
    return np.square(V)

def get_W(A, q):
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

def get_rrmse(original, predicted, epsilon=1e-10):
    return np.linalg.norm(original-predicted) / \
        (np.linalg.norm(original) + epsilon)

def sensitivity_specificity_and_mcc(original, predicted, threshold=0.2):
    original_discrete = (original > 0.2).astype(np.int64)
    predicted_discrete = (predicted > 0.2).astype(np.int64)
    tp = np.sum(np.multiply(original_discrete, predicted_discrete))
    tn = np.sum(np.multiply(1-original_discrete, 1-predicted_discrete))
    fp = np.sum(np.multiply(1-original_discrete, predicted_discrete))
    fn = np.sum(np.multiply(original_discrete, 1-predicted_discrete))
    sensitivity = (tp / (tp + fn)) if tp+fn > 0 else 1
    specificity = (tn / (tn + fp)) if tn+fp > 0 else 1
    mcc_numer = tp*tn - fp*fn
    mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (mcc_numer/mcc_denom) if mcc_denom != 0 else None
    return sensitivity, specificity, mcc

def normalize_weight_vector(weights):
    p = weights.shape[0]
    return weights * math.sqrt(p) / np.linalg.norm(weights)

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