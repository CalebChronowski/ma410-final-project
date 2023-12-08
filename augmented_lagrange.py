import numpy as np

x0 = np.array([1],[-1])
λ0 = -1 # λ
ρ0 = 10 # ρ
tol = 1.e-2 # desired tolerance
sp_tol = 1.e-3 # subproblem tolerance



# f(x) = e^{3*x1} + e^{-4*x2}
def f(x):
    x1, x2 = x[0], x[1]
    return np.e**(3*x1) + np.e**(4*x2)


def grad_f(x):
    x1, x2 = x[0], x[1]
    output = np.array([
        [3*np.e**(3*x1)],
        [-4*np.e**(-4*x2)]
        ])
    return output


def grad2_f(x):
    x1, x2 = x[0], x[1]
    output = np.array([
        [9*np.e**(3*x1), 0],
        [0, 16*np.e**(-4*x2)]
        ])
    return output




# g(x) = x1^2 + x2^2 -1
def g(x):
    x1, x2 = x[0], x[1]
    return x1 + x2 - 1


def grad_g(x):
    x1, x2 = x[0], x[1]
    output = np.array([
        [2*x1],
        [2*x2]
        ])
    

def grad2_g(x):
    np.array([
        [2, 0],
        [0, 2]
        ])



# Lagrange
def L(x, λ):
    return f(x) - λ*g(x)


def grad_L(x, λ):
    return grad_f(x) - λ*grad_g(x)



# Augmented lagrange
def A(f, x, λ, ρ):
    '''
    returns the augmented lagrange function
    '''
    return f(x) + λ*g(x) + 0.5*ρ*g(x)**2


def subproblem(f, x, λ, ρ):
    '''
    an unconstrained subproblem computes x_k+1 by minimizing an augmented lagrange formula
    
    minimize A(x, λ) = f(x) - λ.T * g(x) + 0.5 * ρ g(x).T * g(x)
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