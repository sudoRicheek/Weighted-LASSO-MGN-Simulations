"""
Adithya, Richeek, Aaron, 2022.
This file has the main driver code. Does not provide any further functionality by itself.
"""

import enum
from globals import *
from vars import Generator
from optim import monte_carlo_simulation

import os
import json
import math
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

def main():
    # warnings.filterwarnings("ignore")
    performance_map = {}
    
    if os.path.exists("data/perf-100-new-75.json"):
        performance_map = json.load(open("data/perf-100-new-75.json"))

    n_np = np.array(n_values)
    s_np = np.array(sparsity_values)
    w_cal = np.zeros((len(n_values), len(sparsity_values)))
    for i,n in enumerate(n_values):
        for j,s in enumerate(sparsity_values):
            w_cal[i,j] = float(\
                performance_map[str(n)][str(s)]['extra_info']['best_lambd'])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.meshgrid(n_np, s_np)
    x = x.flatten()
    y = y.flatten()
    z = w_cal.flatten()
    ax.plot_trisurf(x, y, z, cmap=plt.cm.Spectral, edgecolor='none')
    ax.set_xlabel('n')
    ax.set_ylabel('sparsity')
    ax.set_zlabel('lambda')
    ax.set_title('Variation of lambda with n and sparsity (weighted)')
    plt.show()
    
    return
    
    if os.path.exists("data/perf-single-100.json"):
        performance_map = json.load(open("data/perf-single-100.json"))
    
    for n in n_values:
        nstr = str(n)
        if nstr not in performance_map:
            performance_map[nstr] = {}
        for sparsity in sparsity_values:
            sparsitystr = str(sparsity)
            if sparsitystr in performance_map[nstr]:
                print("Found {},{} - skipping".format(n, sparsity))
                continue
            else:
                print("Starting {},{}".format(n, sparsity))
            generator = Generator(n=n, sparsity=sparsity)
            rrmse, sensitivity, specificity, mcc, extra_info = \
                monte_carlo_simulation(generator, num=100, debug=True, \
                    best_lambd=None, single_weight=False)
            performance_map[nstr][sparsitystr] = {
                "rrmse" : rrmse,
                "sensitivity" : sensitivity,
                "specificity" : specificity,
                "mcc" : mcc,
                "extra_info" : extra_info
            }
            json.dump(performance_map, open(\
                "data/perf-100-new-75.json", 'w+'))
    
    """
    print("RRMSE: {}%".format(rrmse*100.0))
    print("Sensitivity: {}".format(sensitivity))
    print("Specificity: {}".format(specificity))
    print("MCC: {}".format(mcc))
    """
    
    """
    num_ns = len(n_values)
    num_ss = len(sparsity_values)
    
    plot = np.zeros((num_ns, num_ss))
    for i, n in enumerate(n_values):
        for j, s in enumerate(sparsity_values):
            plot[i,j] = performance_map[str(n)][str(s)]['specificity']
    
    h = sns.heatmap(plot, cmap=cm.gray, yticklabels=n_values, \
        xticklabels=sparsity_values, annot=True)
    h.set(ylabel='Number of measurements', \
        xlabel='Fraction of people infected', \
        title='Specificity of weighted LASSO')
    plt.savefig('data/plots/specificity-weighted.png')
    """
    
if __name__ == '__main__':
    main()