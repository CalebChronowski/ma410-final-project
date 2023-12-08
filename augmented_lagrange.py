import numpy as np

x0 = np.array([1],[-1])
λ0 = -1 # λ
ρ0 = 10 # ρ
tol = 1.e-2 # desired tolerance
sp_tol = 1.e-3 # subproblem tolerance


def fn(x):
    x1, x2 = x[0], x[1]
    return np.e**(3*x1) + np.e**(4*x2)


grad_fn(x):
    x1, x2 = x[0], x[1]
    
    return


def A_ρ(fn, x, λ, ρ):
    '''
    the augmented lagrange function
    '''



def subproblem(fn, x, λ, ρ):
    '''
    an unconstrained subproblem computes x_k+1 by minimizing an augmented lagrange formula
    
    minimize A(x, λ) = fn(x) - λ.T * g(x) + 0.5 * ρ g(x).T * g(x)
    '''
    


def augmented_lagrange(fn, x0, λ0, ρ0, tol=1e-2, sp_tol=1e-2):
    '''
    performs an augmented lagrange method to solve a nonlinear optimization problem
    
    inputs:
        fn: a function of interes
        x0: an initial guess for x 
        λ0: an initial guess for λ 
        ρ0: an initial guess for ρ
        tol: a specified error tolerance
    
    outputs:
        <UPDATE ME> I haven't decided shape
    '''
    x, λ, ρ = x0, λ0, ρ0
    error = 1
    while error > tol:
        pass


if __name__ == "__main__":
    pass