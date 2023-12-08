# Dependencies
import numpy as np
from logger import Logger


# f(x) = e^{3*x1} + e^{-4*x2}
def f(x):
    x1, x2 = x[0][0], x[1][0] # rewriting the variables this way allows me to make the formula look as close to handwritten math as possible
    return np.e**(3*x1) + np.e**(4*x2)


def grad_f(x):
    x1, x2 = x[0][0], x[1][0]
    output = np.array([
        [3*np.e**(3*x1)],
        [-4*np.e**(-4*x2)]
        ])
    return output


def grad2_f(x):
    x1, x2 = x[0][0], x[1][0]

    output = np.array([
        [9*np.e**(3*x1), 0],
        [0, 16*np.e**(-4*x2)]
        ])
    return output




# g(x) = x1^2 + x2^2 -1
def g(x):
    x1, x2 = x[0][0], x[1][0]
    return x1 + x2 - 1


def grad_g(x):
    x1, x2 = x[0][0], x[1][0]
    output = np.array([
        [2*x1],
        [2*x2]
        ])
    return output
    

def grad2_g(x):
    # x isn't necessary as an argument, but g(x) is how the math would be handwritten and I wanted to preserve that
    output = np.array([
        [2, 0],
        [0, 2]
        ])
    return output


# Lagrange
def L(x, λ): # this is probably useless garbage
    return f(x) - λ*g(x)


# Augmented Lagrange
def A(f, x, λ, ρ):
    '''
    returns the augmented lagrange function
    '''
    return f(x) + λ*g(x) + 0.5*ρ*g(x)**2


def grad_A(f, x, λ, ρ):
    # we're not updating rho apparently
    x1, x2 = x[0][0], x[1][0]
    pd_wrt_x1 = 20*x1*(x1**2 + x2**2 -1) - 2*λ*x1 + 3*np.e**(3*x1)
    pd_wrt_x2 = 20*x2*(x1**2 + x2**2 - 1) - 2*λ*x2 - 4*np.log(3)*(1/81)**(x2) ## this is off by a hundredth
    print(f"pd wrt x1: {pd_wrt_x1}")
    print(f"pd wrt x2: {pd_wrt_x2}")
        
    output = np.array([
        [pd_wrt_x1],
        [pd_wrt_x2]
        ])
    return output


def grad2_A(f, x, λ, ρ):
    x1, x2 = x[0][0], x[1][0]
    pd_wrt_x1x1 = 20*(3*x1**2 + x2**2 - 1)- 2*λ + 9*np.e**(3*x1) # top left
    pd_wrt_x2x2 = 20*(3*x2**2 + x1**2 - 1) - 2*λ + 16*(np.log(3)**2)*(1/81)**(x2) # bottom right
    pd_wrt_x1x2 = 40*x1*x2 
    pd_wrt_x2x1 = pd_wrt_x1x2 # same
    
    output = np.array([
        [pd_wrt_x1x1, pd_wrt_x1x2],
        [pd_wrt_x2x1, pd_wrt_x2x2]
    ])
    return output

###############################################################
# vetigial?
def grad_L(x, λ, ρ):
    return grad_f(x) - λ*grad_g(x)

def grad2_L(x, λ, ρ):
    return grad2_f(x) + λ * grad2_g(x) # verify this
################################################################



# The full Augmented Lagrange method
def subproblem(f, x, λ, ρ):
    '''
    an unconstrained subproblem computes x_k+1 by minimizing an augmented lagrange formula
    
    minimize A(x, λ) = f(x) - λ.T * g(x) + 0.5 * ρ g(x).T * g(x)
    '''
    ### x_k+1 = x_k + (-1 * inv(grad2_L(x)) * grad_L(x) ) ?
    ### thought this was a newton's method problem?
    


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

    data.update({{"initial values"} : {
        "x_0" : x0,
        "lambda_0" : λ0,
        "rho_0" : ρ0,
        "error_tol" : tol,
        "sub_error_tol" : sp_tol
    }})

    data.update({"msg" : "trying something"})
    
    error = 1e8 # set an initial error that's well above any sane tolerance
    while error > tol:
        pass
    data.save()  # overwrites the data.py


if __name__ == "__main__":
    # initial values
    x0 = np.array([
        [-1],
        [1]
        ])
    λ0 = -1
    ρ0 = 10
    tol = 1.e-2 # main error tolerance
    sp_tol = 1.e-3 # subproblem error tolerance ###<Can this be dynamic?>###

    # data
    # this will be accessed to store data for the graphs and other metadata 
    data = Logger

    pass