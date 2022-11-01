"""
Adithya, Richeek, Aaron, 2022.
This file has the main driver code. Does not provide any further functionality by itself.
"""

from globals import *
from vars import Generator
from optim import monte_carlo_simulation

import os
import json
import warnings

def main():
    # warnings.filterwarnings("ignore")
    performance_map = {}
    if os.path.exists("data/perf-single.json"):
        performance_map = json.load(open("data/perf-single.json"))
    
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