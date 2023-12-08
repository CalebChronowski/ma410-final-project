import numpy as np
from augmented_lagrange import *

error_tolerance = 0.001
x_initial = 1
lambda_initial = 1 # λ
rho_initial = 1 # ρ
# lagrange = ℒ

def main(error_tolerance):
    error = 1
    x, _lambda, rho  = x_initial, lambda_initial, rho_initial
    while error > error_tolerance:
        pass
    pass


if __name__ == "__main__":
    main()