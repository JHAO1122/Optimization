import numpy as np
import math
from scipy import sparse
def steepest_descent(f, x0, max_iter, epsilon):
    x_k = np.array(x0, dtype=float)
    trajectory = [x_k.copy()]
    for i in range(max_iter):
        gradf_k = numerical_gradient(f, x_k)
        if np.linalg.norm(gradf_k) < epsilon:
            break
        pk = -gradf_k
        alpha_k = back_searched_step_size(f, x_k, pk, gradf_k, alpha0=0.3)
        x_k = x_k + alpha_k * pk
        trajectory.append(x_k.copy())
    return x_k, trajectory

def back_searched_step_size(f, x_k, pk, gradf_k, alpha0, max_backtrack=1000):
    alpha = alpha0
    rho = 0.5
    c = 1e-4
    for i in range(max_backtrack):
        if f(x_k + alpha * pk) <= f(x_k) + c * alpha * np.dot(gradf_k, pk):
            break
        alpha *= rho
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
def barzilai_borwein(f, x0, max_iter, epsilon, type):
    x_k = np.array(x0, dtype=float)
    trajectory = [x_k.copy()]
    gradf_k = numerical_gradient(f, x_k)
    if np.linalg.norm(gradf_k) < epsilon:
        return x_k, trajectory
    pk = -gradf_k
    alpha_k = back_searched_step_size(f, x_k, pk, gradf_k, alpha0=1.0)
    x_k_prev = x_k.copy()
    x_k = x_k + alpha_k * pk
    trajectory.append(x_k.copy())
    gradf_prev = gradf_k.copy()
    for i in range(max_iter):
        gradf_k = numerical_gradient(f, x_k)
        if np.linalg.norm(gradf_k) < epsilon:
            break
        s_k = x_k - x_k_prev
        y_k = gradf_k - gradf_prev
        if type == 'Short':
            if np.dot(y_k, y_k) > 1e-15:
                alpha_k = np.dot(s_k, y_k) / np.dot(y_k, y_k)
            else: 
                alpha_k = 1.0
        elif type == 'Long':
            if np.dot(s_k, y_k) > 1e-15:
                alpha_k = np.dot(s_k, s_k) / np.dot(s_k, y_k)
            else:
                alpha_k = 1.0
        else:
            alpha_k = 1.0
        p_k = -gradf_k
        x_k_prev = x_k.copy()
        x_k = x_k + alpha_k * p_k
        gradf_prev = gradf_k.copy()
        trajectory.append(x_k.copy())
    return x_k, trajectory

def newton_method(f, x0, max_iter, epsilon, type):
    x_k = np.array(x0, dtype=float)
    trajectory = [x_k.copy()]
    for i in range(max_iter):
        gradf_k = numerical_gradient(f, x_k)
        if np.linalg.norm(gradf_k) < epsilon:
            break
        Hk = numerical_hessian(f, x_k)
        
        pk = -np.linalg.solve(Hk, gradf_k)
        if type == 'pure':
            alpha_k = 1.0
        elif type == 'line_search':
            alpha_k = back_searched_step_size(f, x_k, pk, gradf_k, alpha0=1.0)
        else:
            raise ValueError("Invalid type of Newton method")
        x_k = x_k + alpha_k * pk
        trajectory.append(x_k.copy())
    return x_k, trajectory

def analyze_convergence(trajectory, f, op_value, method_name=''):
    print(f"\n{method_name} 收敛分析:")
    f_values = [f(x) for x in trajectory]
    op_gap = [abs(f_val - op_value) for f_val in f_values]
    if len(op_gap) > 0 and op_gap[-1] == 0:
        op_gap = op_gap[:-1]
    convergence_order = []
    for i in range(2, len(op_gap)):
        if op_gap[i] > 1e-15 and op_gap[i-1] > 1e-15 and op_gap[i-2] > 1e-15:
            numerator = np.log(op_gap[i] / op_gap[i-1])
            denominator = np.log(op_gap[i-1] / op_gap[i-2])
            if abs(denominator) > 1e-10:  
                order = numerator / denominator
                convergence_order.append(order)
    if convergence_order:
        avg_order = np.mean(convergence_order[-5:]) 
        print(f"  估计收敛阶数 (Q-alpha): {avg_order:.4f}")
        if avg_order > 1.7:
            print("  收敛类型: Q-二次收敛")
        elif avg_order > 1.2:
            print("  收敛类型: Q-超线性收敛")
        elif avg_order > 0.8:
            print("  收敛类型: Q-线性收敛")
        else:
            print("  收敛类型: Q-次线性收敛")
    return convergence_order
    
    if convergence_order:
        avg_order = np.mean(convergence_order[-5:])  
        print(f"  估计收敛阶数 (Q-alpha): {avg_order:.4f}")
        if avg_order > 1.8:
            print("  收敛类型: Q-二次收敛")
        elif avg_order > 1.2:
            print("  收敛类型: Q-超线性收敛")
        elif avg_order > 0.8:
            print("  收敛类型: Q-线性收敛")
        else:
            print("  收敛类型: Q-次线性收敛")
    return convergence_order


def numerical_hessian(f, x, h=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        H[i, i] = (f(x_plus) - 2*f(x) + f(x_minus)) / (h**2)
    for i in range(n):
        for j in range(i+1, n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
            H[j, i] = H[i, j]
    return H

def f_five(x, A, b, miu, delta):
    data_fitting = 0.5 * np.sum((A @ x - b)**2)
    L_delta = 0
    for i in range(len(x)):
        L_delta += (x[i]**2) / (2 * delta)
    else:
        L_delta += abs(x[i]) - 0.5 * delta
    return data_fitting + miu * L_delta


def main():
    f_rosenbrock = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    f_beale = lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
    x0 = np.array([-1.2, 1.0])
    x1 = np.array([2.0, 0.0])
    x_newton, trajectory_newton_ro = newton_method(f_rosenbrock, x0, 10000, 1e-6, 'line_search')
    print("Newton's method of Rosenbrock function: x =", x_newton)
    x_steepest, trajectory_steepest_ro = steepest_descent(f_rosenbrock, x0, 10000, 1e-6)
    print("Steepest descent of Rosenbrock function: x =", x_steepest)
    x_beale, trajectory_newton_beale = newton_method(f_beale, x1, 10000, 1e-6, 'line_search')
    print("Newton's method of Beale function: x =", x_beale)
    x_beale, trajectory_steepest_beale = steepest_descent(f_beale, x1, 10000, 1e-6)
    print("Steepest descent of Beale function: x =", x_beale)
    A = np.array([
        [5, 1, 0, 0.5],
        [1, 4, 0.5, 0],
        [0, 0.5, 3, 0],
        [0.5, 0, 0, 2]
    ])
    def f(x, sigma):
        x = np.array(x)
        return 0.5 * np.dot(x, x) + 0.25 * sigma * (np.dot(x, np.dot(A, x)) ** 2 )
    x_four_one = np.array([math.cos(7* math.pi / 18), math.sin(7* math.pi / 18), math.cos(7* math.pi / 18), math.sin(7* math.pi / 18)])
    x_four_two = np.array([math.cos(5* math.pi / 18), math.sin(5* math.pi / 18), math.cos(5* math.pi / 18), math.sin(5* math.pi / 18)])
    x_four_pure_newton_sigma_1, trajectory_newton_four_one_pure_sigma_1 = newton_method(lambda x: f(x, 1), x_four_one, 10000, 1e-6, 'pure')
    print("Pure Newton's method of Four-one function with sigma=1: x =", x_four_pure_newton_sigma_1)
    x_four_line_newton_sigma_1, trajectory_newton_four_one_line_sigma_1 = newton_method(lambda x: f(x, 1), x_four_one, 10000, 1e-6, 'line_search')
    print("Line search Newton's method of Four-one function with sigma=1: x =", x_four_line_newton_sigma_1)
    x_four_pure_newton_sigma_1e4, trajectory_newton_four_one_pure_sigma_1e4 = newton_method(lambda x: f(x, 1e4), x_four_one, 10000, 1e-6, 'pure')
    print("Pure Newton's method of Four-one function with sigma=10000: x =", x_four_pure_newton_sigma_1e4)
    x_four_line_newton_sigma_1e4, trajectory_newton_four_one_line_sigma_1e4 = newton_method(lambda x: f(x, 1e4), x_four_one, 10000, 1e-6, 'line_search')
    print("Line search Newton's method of Four-one function with sigma=10000: x =", x_four_line_newton_sigma_1e4)
    x_four_pure_newton_sigma_1_two, trajectory_newton_four_two_pure_sigma_1= newton_method(lambda x: f(x, 1), x_four_two, 10000, 1e-6, 'pure')
    print("Pure Newton's method of Four-two function with sigma=1: x =", x_four_pure_newton_sigma_1_two)
    x_four_line_newton_sigma_1_two, trajectory_newton_four_two_line_sigma_1 = newton_method(lambda x: f(x, 1), x_four_two, 10000, 1e-6, 'line_search')
    print("Line search Newton's method of Four-two function with sigma=1: x =", x_four_line_newton_sigma_1_two)
    x_four_pure_newton_sigma_1e4_two, trajectory_newton_four_two_pure_sigma_1e4 = newton_method(lambda x: f(x, 1e4), x_four_two, 10000, 1e-6, 'pure')
    print("Pure Newton's method of Four-two function with sigma=10000: x =", x_four_pure_newton_sigma_1e4_two)
    x_four_line_newton_sigma_1e4_two, trajectory_newton_four_two_line_sigma_1e4 = newton_method(lambda x: f(x, 1e4), x_four_two, 10000, 1e-6, 'line_search')
    print("Line search Newton's method of Four-two function with sigma=10000: x =", x_four_line_newton_sigma_1e4_two)

    analyze_convergence(trajectory_newton_four_one_pure_sigma_1, lambda x: f(x, 1), 0, 'Newton\'s method of question four function in pure mode\n')
    analyze_convergence(trajectory_newton_four_one_line_sigma_1, lambda x: f(x, 1), 0, 'Newton\'s method of question four function in line search mode\n')
    analyze_convergence(trajectory_newton_four_one_pure_sigma_1e4, lambda x: f(x, 1e4), 0, 'Newton\'s method of question four function in pure mode\n')
    analyze_convergence(trajectory_newton_four_one_line_sigma_1e4, lambda x: f(x, 1e4), 0, 'Newton\'s method of question four function in line search mode\n')
    analyze_convergence(trajectory_newton_four_two_pure_sigma_1, lambda x: f(x, 1), 0, 'Newton\'s method of question four function in pure mode\n')
    analyze_convergence(trajectory_newton_four_two_line_sigma_1, lambda x: f(x, 1), 0, 'Newton\'s method of question four function in line search mode\n')
    analyze_convergence(trajectory_newton_four_two_pure_sigma_1e4, lambda x: f(x, 1e4), 0, 'Newton\'s method of question four function in pure mode\n')
    analyze_convergence(trajectory_newton_four_two_line_sigma_1e4, lambda x: f(x, 1e4), 0, 'Newton\'s method of question four function in line search mode\n')
    
    
    m, n = 512, 1024
    r = 0.1
    A = np.random.randn(m, n)
    x_true_sparse = sparse.random(n, 1, density=r, random_state=42)
    x_true = x_true_sparse.toarray().flatten()
    b = A @ x_true
    for miu in [1e-2, 1e-3]:
        print(f"\n=== 使用 μ = {miu} 求解问题5 ===")
        delta = 1e-3 * miu
        f_obj = lambda x: f_five(x, A, b, miu, delta)
        x0_five = np.random.randn(n)  
        x_opt, trajectory = steepest_descent(f_obj, x0_five, max_iter=10, epsilon=1e-6)
        print(f"Steepest descent method of problem 5 with μ = {miu}: x = {x_opt}")
        x_opt_barzilai, trajectory = barzilai_borwein(f_obj, x0_five, max_iter=10, epsilon=1e-6, type='Short')
        print(f"Barzilai-Borwein method of problem 5 with μ = {miu} and type = Short: x = {x_opt_barzilai}")
        x_opt_barzilai, trajectory = barzilai_borwein(f_obj, x0_five, max_iter=10, epsilon=1e-6, type='Long')
        print(f"Barzilai-Borwein method of problem 5 with μ = {miu} and type = Long: x = {x_opt_barzilai}")
if __name__ == '__main__':
    main()  