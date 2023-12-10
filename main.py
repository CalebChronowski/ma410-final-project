import numpy as np
from augmented_lagrange import *

if __name__ == "__main__":
    # initial values
    x0 = np.array([
        [-1],
        [1]
        ])
    λ0 = -1
    ρ0 = 10
    x = augmented_lagrange(x0, λ0, ρ0)

    print(f"x = {x}")
