import numpy as np
import math
import matplotlib.pyplot as plt
def extended_rosenbrock(x):
    n = len(x)
    f_val = 0.0
    for i in range(n-1):
        f_val += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return f_val

def powell_singular(x):
    return (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4
    
def main():
    plot_convergence()
    n_values = [5, 8, 12, 20]
    with open('cg_results.txt', 'w') as f:
        f.write("CG Method Results\n")
        f.write("=================\n\n")
        for n in n_values:
            A = generate_A(n)
            b = np.ones(n)
            x0 = np.zeros(n)
            k, residuals = CG(A, b, x0)
            f.write(f"n = {n}: {k} iterations, final residual = {residuals[-1]:.2e}\n")
    print("BFGS Optimization Results:")
    tol = 1e-5
    print("\n1. Extended Rosenbrock Function")
    n_values_four = [6, 8, 10]
    for n in n_values_four:
        x0 =np.zeros(n)
        for i in range(n):
            if i % 2 == 0:
                x0[i] = -1.2
            else:
                x0[i] = 1.0
        print(f"\nDimension n = {n}:")
        print(f"Initial point: {x0}")
        print(f"Target point: [1, 1, ..., 1]")
        x_opt, iterations, grad_norm = BFGS(extended_rosenbrock, x0, n, tol=tol)
        print(f"Optimized solution: {x_opt}")
        print(f"Function value at solution: {extended_rosenbrock(x_opt):.6e}")
        print(f"Number of iterations: {iterations}")
        print(f"Gradient norm at solution: {grad_norm:.6e}")
        print(f"Distance to true optimum: {np.linalg.norm(x_opt - np.ones(n)):.6e}")
    print("\n\n2. Powell Singular Function")
    n_powell = 4
    x0_powell = np.array([3.0, -1.0, 0.0, 1.0])
    x_target_powell = np.array([0.0, 0.0, 0.0, 0.0])
    x_opt_powell, iterations_powell, grad_norm_powell = BFGS(powell_singular, x0_powell, n_powell, tol=tol)
    print(f"Optimized solution: {x_opt_powell}")
    print(f"Function value at solution: {powell_singular(x_opt_powell):.6e}")
    print(f"Number of iterations: {iterations_powell}")
    print(f"Gradient norm at solution: {grad_norm_powell:.6e}")
    print(f"Distance to true optimum: {np.linalg.norm(x_opt_powell - x_target_powell):.6e}")


def BFGS(f ,x0, n, tol=1e-6, max_iter=1000):
    k = 0
    H0 = np.eye(n)
    g0 = numerical_gradient(f, x0)
    while (k < max_iter and np.linalg.norm(g0) > tol):
        p0 = -H0 @ g0
        alpha0 = wolfe(f, x0, p0, g0, 1.0)
        x1 = x0 + alpha0*p0
        s0 = x1 - x0
        g1 = numerical_gradient(f, x1)
        y0 = g1 - g0
        if y0 @ s0 <= 1e-12:
            H1 = np.eye(n)
        else:
            pho = 1.0 / (y0 @ s0)
            H1 = (np.eye(n) - pho * np.outer(s0, y0)) @ H0 @ (np.eye(n) - pho * np.outer(y0, s0)) + pho * np.outer(s0, s0)
        x0 = x1
        g0 = g1
        H0 = H1
        k += 1
    return x0, k, np.linalg.norm(g0)

        

def wolfe(f, x_k, pk, gradf_k, alpha0, max_backtrack=100):
    alpha = alpha0
    c1 = 1e-4
    c2 = 0.9
    for i in range(max_backtrack):
        x_new = x_k + alpha * pk
        f_new = f(x_new)
        f_k = f(x_k)
        if f_new > f_k + c1 * alpha * np.dot(gradf_k, pk):
            alpha *= 0.5
            continue  
        grad_new = numerical_gradient(f, x_new)
        if np.dot(grad_new, pk) < c2 * np.dot(gradf_k, pk):
            alpha *= 1.5
        else:
            return alpha
            
    return alpha


def numerical_gradient(f, x):
    h = 1e-4
    gradf = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        gradf[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return gradf

def CG(A, b, x0, tol=1e-6, max_iter=1000):
    r0 = A @ x0 - b
    p0 = -r0
    k = 0
    residuals = [np.linalg.norm(r0)] 
    while (np.linalg.norm(r0) > tol and k < max_iter):
        alpha0 = r0 @ r0 / (p0 @ A @ p0)
        x1 = x0 + alpha0 * p0
        r1 = r0 + alpha0 * A @ p0
        beta1 = (r1 @ r1) / (r0 @ r0)
        p1 = -r1 + beta1 * p0
        x0 = x1
        r0 = r1
        p0 = p1
        k += 1
        residuals.append(np.linalg.norm(r0))
        
    return k, residuals

def generate_A(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i][j] = 1.0 / (i + j + 1)
    return A
    
def plot_convergence():
    n_values = [5, 8, 12, 20]   
    plt.figure(figsize=(12,8))
    for n in n_values:
        A = generate_A(n)
        b = np.ones(n)
        x0 = np.zeros(n)
        iterations, residuals = CG(A, b, x0)
        plt.semilogy(range(len(residuals)), residuals, 
                    marker='o', markersize=3, linewidth=2, 
                    label=f"n = {n}, CG iterations = {iterations}")
        if residuals[-1] < 1e-6:
            plt.plot(len(residuals)-1, residuals[-1], 'r*', markersize=10)
    plt.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, 
                label=' $10^{-6}$')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Convergence of Conjugate Gradient')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-8, 10)
    plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight') 
        
if __name__ == "__main__":
    main()    


        