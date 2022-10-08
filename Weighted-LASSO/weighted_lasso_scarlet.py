"""
Generate the metrics for 50 days for a given simulation data
and a given sensing matrix.
"""
import json

import cvxpy as cp

import numpy as np
from numpy import linalg
from numpy.random import default_rng

from scipy.io import loadmat

import matplotlib.pyplot as plt

from utils import lasso, weighted_lasso, cross_validation

rng = default_rng(42)
M = 375
N = 1000
tau = 0.001 # weight for lasso
sigma = 0.001 # Noise in measurement

EPSILON = 1e-8


"""
data10 -> 3.29
data7 -> 8
data6 -> 6
data5 -> 4.22
data1 -> 8.76
[10,5,6,1]
"""

DATA_NO = 10
DATA_FILE = f"data/ct_data_general{DATA_NO}.mat"
# DATA_FILE = f"../Ritesh-Code/Infection Spread Model/Data/ct_data_general_{DATA_NO}.mat"

SENSING_MATRIX_FILE = f"../Ritesh-Code/Pooling Matrix Design/Kirkman Matrices/Examples/kirkman_{M}_{N}.txt"

OUTPUT_DUMP = f"out/simulation_{M}_{DATA_NO}.json"

load_data = loadmat(DATA_FILE)
TOTAL_DAYS = load_data['K'][0][0]

num_infected = [np.count_nonzero(load_data['X'][i][0]) for i in range(TOTAL_DAYS)]
peak_day = np.argmax(num_infected)
sparsity = (np.mean(num_infected[peak_day-24:peak_day+25])/N)*100
# print(sparsity)

# Plot the number of infected people vs time
PLOT_INFECTED = False
if PLOT_INFECTED:
    plt.plot(range(1, TOTAL_DAYS+1), num_infected, linestyle="-", linewidth=2, color='r', alpha=0.6, label='# infected people')
    plt.title("Number of infected people over time", fontsize=18)
    plt.xlabel("Days", fontsize=15) 
    plt.ylabel("Number of Infected", fontsize=15) 
    plt.legend()
    plt.show()

TEST_DAYS = np.arange(peak_day-25, peak_day+25)

TABULATE = {
    'M': M,
    'N': N,
    'Data-No': DATA_NO,
    'Sparsity': sparsity,
    'P-LASSO': {
        'FP': [],
        'FN': [],
        'TP': [],
        'TN': [],
        'RRMSE': []
    },
    'W-LASSO': {
        'FP': [],
        'FN': [],
        'TP': [],
        'TN': [],
        'RRMSE': []
    },
}


for day in TEST_DAYS:
    Pr = load_data['Pr'][day][0].reshape(N,)
    X = load_data['X'][day][0].reshape(N,)
    print("Pr:", Pr.shape)
    print("X:", X.shape)


    # LOAD SENSING MATRIX
    if SENSING_MATRIX_FILE[-3:] == "txt":
        A = np.loadtxt(SENSING_MATRIX_FILE, delimiter=',')
    elif SENSING_MATRIX_FILE[-3:] == "mat":
        A = loadmat(SENSING_MATRIX_FILE)['B'].todense()
    else:
        print("NOT A VALID MATRIX FILE. EXITING...")
        exit(1)
    print("A:", A.shape)


    # GENERATE THE NOISY MEASUREMENTS
    noise = rng.normal(loc=0, scale=sigma, size=A.shape[0])
    q = 0.95
    Y = (A@X)*(1 + q)**(noise)
    print("Y:", Y.shape)


    # PLAIN LASSO
    X_recons_pl = cp.Variable(N)
    lambd_pl = cp.Parameter(nonneg=True)
    constraints_pl = [0 <= X_recons_pl]
    objective_pl = cp.Minimize(lasso.objective_fn(A, Y, X_recons_pl, lambd_pl))
    plain_lasso_problem = cp.Problem(objective_pl, constraints_pl)

    lambd_pl.value = cross_validation(M, N, A, Y, method='plain-lasso')
    print("PLAIN LASSO CV LAMBDA:", lambd_pl.value)


    # WEIGHTED LASSO
    W = np.diag(-1*np.log(Pr + EPSILON))

    X_recons_wl = cp.Variable(N)
    lambd_wl = cp.Parameter(nonneg=True)
    constraints_wl = [0 <= X_recons_wl]
    objective_wl = cp.Minimize(weighted_lasso.objective_fn(A, Y, X_recons_wl, lambd_wl, W))
    weighted_lasso_problem = cp.Problem(objective_wl, constraints_wl)

    lambd_wl.value = cross_validation(M, N, A, Y, Pr, method='weighted-lasso')
    print("WEIGHTED LASSO CV LAMBDA:", lambd_wl.value)

    # SOLVE PROBLEMS
    plain_lasso_problem.solve()
    weighted_lasso_problem.solve()

    # RRMSE MEASURES
    print("RRMSE Plain LASSO:", linalg.norm(X-X_recons_pl.value)/linalg.norm(X))
    print("RRMSE Weighted LASSO:", linalg.norm(X-X_recons_wl.value)/linalg.norm(X))

    # SPECIFICITY AND SENSITIVITY MEASURES
    tp = np.count_nonzero(X)
    tn = N - tp
    # sparsity = (tp/N)*100

    #### choice of threshold = 0.2, Goenka et. al.
    test_sample = (X > 0.2).astype(int)
    result_pl = (X_recons_pl.value > 0.2).astype(int)
    result_wl = (X_recons_wl.value > 0.2).astype(int)

    
    fn_pl = ((test_sample - result_pl)==1).sum()
    fp_pl = ((test_sample - result_pl)==-1).sum()
    fn_wl = ((test_sample - result_wl)==1).sum()
    fp_wl = ((test_sample - result_wl)==-1).sum()

    print(f'Sparsity: {sparsity}' + '%')
    print(f'FN PL: {fn_pl}')
    print(f'FP PL: {fp_pl}')
    print(f'FN WL: {fn_wl}')
    print(f'FP WL: {fp_wl}')

    TABULATE['P-LASSO']['FP'] += [int(fp_pl)]
    TABULATE['P-LASSO']['FN'] += [int(fn_pl)]
    TABULATE['P-LASSO']['TP'] += [tp]
    TABULATE['P-LASSO']['TN'] += [tn]
    TABULATE['P-LASSO']['RRMSE'] += [linalg.norm(X-X_recons_pl.value)/linalg.norm(X)]

    TABULATE['W-LASSO']['FP'] += [int(fp_wl)]
    TABULATE['W-LASSO']['FN'] += [int(fn_wl)]
    TABULATE['W-LASSO']['TP'] += [tp]
    TABULATE['W-LASSO']['TN'] += [tn]
    TABULATE['W-LASSO']['RRMSE'] += [linalg.norm(X-X_recons_wl.value)/linalg.norm(X)]


print(TABULATE)
with open(OUTPUT_DUMP, 'w') as fp:
    json.dump(TABULATE, fp)