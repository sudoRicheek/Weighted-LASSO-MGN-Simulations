"""
Adithya, Richeek, Aaron, 2022.
This file has the main driver code. Does not provide any further functionality by itself.
"""

from globals import *
from vars import Generator
from optim import monte_carlo_simulation

import warnings

def main():
    # warnings.filterwarnings("ignore")
    generator = Generator()
    mean, std = monte_carlo_simulation(generator, num=100, debug=True)
    print("Mean Relative RMSE is {}".format(mean))
    print("Standard Deviation in Relative RMSE is {}".format(std))
    
if __name__ == '__main__':
    main()