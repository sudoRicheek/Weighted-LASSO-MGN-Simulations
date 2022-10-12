"""
Adithya, Richeek, Aaron, 2022.
Provides helper functions to perform LASSO and Weighted LASSO.
While some of this functionality exists in Ritesh's code, having an easy one-stop plug-and-play
outlet for the optimization that we are familiar with will be a great help.
"""

from globals import *
import cvxpy as cp
import numpy as np

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

def get_weighted_lasso_objective(x, y, A, w, lambd=1):
    """
    The objective in weighted LASSO is
    || y - Ax ||^2 + lambd \sum_{i=1}^p |w_i x_i|
    Parameters:
    @x - The cvxpy variable representing x
    @Y - n x 1 result vector
    @A - n x p sensing matrix
    @w - p x 1 matrix to be multiplied pre-L1-norm.
    @lambd - scalar regularization parameter
    """
    l2_term = cp.norm2(y - np.matmul(A,x))**2
    regularizer = cp.norm1(np.multiply(w, x))*lambd
    return (l2_term + regularizer)
    
def get_lasso_objective(x, y, A, lambd=1):
    """
    The weighted lasso objective, with all weights equalling 1.
    """
    w = np.ones(A.shape[-1])
    return get_weighted_lasso_objective(x, y, A, w, lambd)

def solve_weighted_lasso(y, A, w=None, lambd=1):
    """
    Performs the weighted lasso optimization. Returns the optimal x found.
    """
    x = cp.Variable(A.shape[-1])
    if w is None:
        objective = get_lasso_objective(x, y, A, lambd)
    else:
        objective = get_weighted_lasso_objective(x, y, A, w, lambd)
    problem = cp.Problem(cp.Minimize(objective))    
    problem.solve()
    return x.value

def monte_carlo_simulation(generator, w=None, lambd=1, num=15000):
    """
    Simulate weighted lasso with the given w and lambda. The generator is
    expected to expose a next() method returning appropriately generated
    (x_real,y,A) triples. The mean and variance of MSE(x_real,x) across the runs is
    returned.
    The variance here does not include Bessel's correction
    """
    msum = 0
    msqsum = 0
    for _ in range(num):
        x_real, y, A = generator.next()
        x = solve_weighted_lasso(y, A, w=w, lambd=lambd)
        mse = np.linalg.norm(x-x_real)**2 / x.shape[0]
        msum += mse
        msqsum += mse**2
    mean = msum / num
    var = (msqsum/num) - mean**2
    return mean, var

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