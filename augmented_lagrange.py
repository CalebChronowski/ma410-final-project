# Dependencies
import numpy as np

### Functions and gradients ###
# g(x) = x1^2 + x2^2 -1
def g(x):
    x1, x2 = x[0][0], x[1][0]
    return x1**2 + x2**2 - 1


# augmented lagrange gradients
def grad_A(x, λ):
    x1, x2 = x[0][0], x[1][0]
    pd_wrt_x1 = 20*x1*(x1**2 + x2**2 -1) - 2*λ*x1 + 3*np.e**(3*x1)
    pd_wrt_x2 = 20*x2*(x1**2 + x2**2 - 1) - 2*λ*x2 - 4*np.log(3)*(1/81)**(x2)
        
    output = np.array([
        [pd_wrt_x1],
        [pd_wrt_x2]
        ])
    return output


def grad2_A(x, λ):
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


### The unconstrained subproblem ###
def subproblem(x, λ, sp_tol=1e-9):
    '''
    an unconstrained subproblem computes x_k+1 by minimizing an augmented lagrange formula
    
    minimize A(x, λ) = f(x) - λ.T * g(x) + 0.5 * ρ g(x).T * g(x)
    '''
    x_k = x
    error = 1
    i = 0
    while error > sp_tol:
        gradA = grad_A(x_k, λ)
        grad2A = grad2_A(x_k, λ)
        x_kp1 = x_k - np.matmul(np.linalg.inv(grad2A), gradA)

        # update
        x_k = x_kp1
        i+=1
        error = np.linalg.norm(gradA)
        if i >= 1000000:
            break
    return x_k


### The full augmented lagrange method ###
def augmented_lagrange(x0, λ0, ρ0, tol=1e-9, sp_tol=1e-9):
    '''
    performs an augmented lagrange method to solve a nonlinear optimization problem
    '''
    x, λ, ρ = x0, λ0, ρ0
    error = 1e7
    i = 0

    x_k = x
    λ_k = λ


    while error > tol:
        i+=1
        x_kp1 = subproblem(x_k, λ_k, sp_tol=sp_tol)
        λ_kp1 = λ_k - ρ*g(x_kp1) # lambda update
        
        # update
        x_k = x_kp1
        λ_k = λ_kp1
        error = np.linalg.norm(grad_A(x_k, λ_k))
        print(f"{x_k[0][0]}, {x_k[1][0]}, {λ_k}, {error}")
    return x_k


if __name__ == "__main__":
    # initial values
    x0 = np.array([
        [-1],
        [1]
        ])
    λ0 = -1
    ρ0 = 10

    x = augmented_lagrange(x0, λ0, ρ0)
    print(f"final answer:\n{x}")

    error = np.linalg.norm(grad_A(x0, λ0))
    print(error)