"""
Adithya, Richeek, Aaron, 2022.
Defines the various helpers that are used throughout to get variable values. Also
has helpers to perform re-centering, etc.
"""

from globals import *
import numpy as np
import scipy.io
import math

class Generator:
    def __init__(self, n=n_default, p=p_default, q=q_bernoulli, sparsity=sparsity_default, \
        sigma=gaussian_sigma, approx=True):
        self.n = n
        self.p = p
        self.q = q
        self.sigma = sigma
        self.sparsity = sparsity
        self.approx = approx
    
    def next(self):
        x_real = get_x(self.p, self.sparsity)
        A = get_A(self.n, self.p, self.q)  
        y = get_y(A, x_real, self.sigma, self.approx)
        return x_real, y, A
        

def get_A(n=n_default, p=p_default, q=q_bernoulli):
    return np.random.binomial(1, q, size=(n,p))

def get_x(p=p_default, sparsity=sparsity_default):
    x = np.random.binomial(1, sparsity, size=p)
    x_factor1 = np.random.uniform(0.2, 1000, size=p)
    # x_factor1 = np.random.uniform(0.2, 2**15, size=p)
    # x_factor1 = 1 + np.random.exponential(scale=4.6e-4, size=p)
    x_factor2 = np.random.uniform(0, 0.2, size=p)
    x = ((x == 1).astype(float)*x_factor1) + ((x == 0).astype(float)*x_factor2)
    return x

def add_mult_gaussian_noise(y_before, sigma, approx=True):
    eta = sigma*math.log(1+qa)*np.random.randn(*y_before.shape)
    if approx:
        return np.multiply(y_before, (1+eta))
    else:
        return np.multiply(y_before, np.exp(eta))

def get_y(A, x, sigma=gaussian_sigma, approx=True):
    return add_mult_gaussian_noise(np.matmul(A,x), sigma, approx=approx)

def rescale_and_center(A, y, q=q_bernoulli):
    n =  A.shape[-2]
    Ahat = (A - q) / math.sqrt(n*q*(1-q))
    yhat = (n*y - np.sum(y))/((n-1)*math.sqrt(n*q*(1-q)))
    return Ahat, yhat

if __name__ == '__main':
    pass