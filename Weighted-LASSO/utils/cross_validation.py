import cvxpy as cp

import numpy as np
from numpy import linalg

import matplotlib.pyplot as plt

from utils import lasso, weighted_lasso

EPSILON = 1e-8
LAMBD_VALUES = np.logspace(-2, 1, 50)

def cross_validation(M, N, A, Y, Pr=None, TRAIN_SPLIT=0.8, method='plain-lasso'):
    """
    Methods are: plain-lasso OR weighted-lasso
    """
    M_train = int(M*TRAIN_SPLIT)
    Y_train = Y[:M_train]
    Y_val = Y[M_train:]
    A_train = A[:M_train]
    A_val = A[M_train:]

    if method == 'plain-lasso':
        X_search = cp.Variable(N)
        lambd = cp.Parameter(nonneg=True)
        constraints = [0 <= X_search]
        objective = cp.Minimize(lasso.objective_fn(A_train, Y_train, X_search, lambd))
        problem = cp.Problem(objective, constraints)

        err = []
        for v in LAMBD_VALUES:
            lambd.value = v
            try:
                problem.solve()
                Y_recon = A_val@X_search.value
                err += [linalg.norm(Y_recon - Y_val)**2]
            except cp.error.SolverError:
                err += [float('inf')]
        # print(err)
        # plt.plot(LAMBD_VALUES, err)
        # plt.show()
        return LAMBD_VALUES[np.argmin(err)]
    
    elif method == 'weighted-lasso':    
        if Pr is None:
            print("NO PROBABILITIES SPECIFIED. EXITING...")
            exit(1)
        W = np.diag(-1*np.log(Pr+EPSILON))

        X_search = cp.Variable(N)
        lambd = cp.Parameter(nonneg=True)
        constraints = [0 <= X_search]
        objective = cp.Minimize(weighted_lasso.objective_fn(A_train, Y_train, X_search, lambd, W))
        problem = cp.Problem(objective, constraints)

        err = []
        for v in LAMBD_VALUES:
            lambd.value = v
            try:
                problem.solve()
                Y_recon = A_val@X_search.value
                err += [linalg.norm(Y_recon - Y_val)**2]
            except cp.error.SolverError:
                err += [float('inf')]
        # print(err)
        # plt.plot(LAMBD_VALUES, err)
        # plt.show()
        return LAMBD_VALUES[np.argmin(err)]
