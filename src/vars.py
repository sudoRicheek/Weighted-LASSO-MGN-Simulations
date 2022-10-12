"""
Adithya, Richeek, Aaron, 2022.
Defines the various helpers that are used throughout to get variable values. Also
has helpers to perform re-centering, etc.
"""

from globals import *
import numpy as np
import math

def get_A(n, p, q=q_bernoulli):
    return np.random.binomial(1, q, size=(n,p))

def get_x(p):
    """
    TODO: what distribution do viral loads satisfy?
    """
    return None

def add_mult_gaussian_noise(y_before, sigma, approx=True):
    eta = sigma*math.log(1+qa)*np.random.randn(y_before.shape)
    if approx:
        return np.multiply(y_before, (1+eta))
    else:
        return np.multiply(y_before, np.exp(eta))

def get_y(A, x, sigma=gaussian_sigma, approx=True):
    return add_mult_gaussian_noise(np.multiply(A,x), sigma, approx=approx)

def rescale_and_center(A, y, q=q_bernoulli):
    n =  A.shape[-2]
    Ahat = (A - q) / math.sqrt(n*q*(1-q))
    yhat = (n*y - np.sum(y))/((n-1)*math.sqrt(n*q*(1-q)))
    return Ahat, yhat

if __name__ == '__main':
    pass