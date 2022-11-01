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

def main():
    # warnings.filterwarnings("ignore")
    performance_map = {}
    if os.path.exists("data/perf-single.json"):
        performance_map = json.load(open("data/perf-single.json"))

    num_ns = len(n_values)
    num_ss = len(sparsity_values)
    
    plot = np.zeros((num_ns, num_ss))
    for i, n in enumerate(n_values):
        for j, s in enumerate(sparsity_values):
            plot[i,j] = performance_map[str(n)][str(s)]['rrmse']*math.sqrt(1000)
    
    h = sns.heatmap(plot, yticklabels=n_values, xticklabels=sparsity_values, \
        annot=True)
    h.set(ylabel='Number of measurements', \
        xlabel='Fraction of people infected', \
        title='RMSE of unweighted LASSO')
    plt.savefig('data/plots/rmse-unweighted.png')
    
    return
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
            rrmse, sensitivity, specificity, mcc = monte_carlo_simulation(generator, num=10, debug=True)
            performance_map[nstr][sparsitystr] = {
                "rrmse" : rrmse,
                "sensitivity" : sensitivity,
                "specificity" : specificity,
                "mcc" : mcc
            }
            json.dump(performance_map, open("data/perf-single.json", 'w+'))
    
    """print("RRMSE: {}%".format(rrmse*100.0))
    print("Sensitivity: {}".format(sensitivity))
    print("Specificity: {}".format(specificity))
    print("MCC: {}".format(mcc))"""
    
if __name__ == '__main__':
    main()