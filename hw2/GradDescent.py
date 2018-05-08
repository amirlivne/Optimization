import numpy as np
from scipy.optimize import approx_fprime, rosen
import matplotlib.pyplot as plt

# f_x = rosen(x)
def rosenbrok_func(x):
    if len(x) == 1:
        print('error, vector must be at lease with 2 variables')
        return None
    else:
        return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def armijo(x, func, d, sigma=0.25, beta=0.5):
    """
    Armijo line search
    :param x: point
    :param func: a function to optimize
    :param d: direction of line for the line search
    :param sigma:
    :param beta: divide factor to find optimal alfa
    :return: alfa as float - the optimal step size
    """

    # init alfa
    alfa = 1

    # compute c (the diffrencial of phi(alfa) at alfa=0)
    grad_x = approx_fprime(x, func, epsilon=1e-8)
    c = np.dot(grad_x, d)

    while (func(x + alfa*d) - func(x)) > sigma*c*alfa:
        alfa *= beta

    return alfa


def grad_descent(func, x_init, eps=10e-5, max_iter=400):
    """

    :param func:
    :param x_init:
    :param eps:
    :param max_iter:
    :return:
    """
    x = x_init
    grad_x = approx_fprime(x, func, epsilon=1e-8)
    d = -grad_x
    converge_vals = []
    for i in range(max_iter):
        converge_vals.append(func(x))
        print(np.linalg.norm(grad_x))
        if np.linalg.norm(grad_x) < eps:
            break
        else:
            x += armijo(x, func, d)*d
            grad_x = approx_fprime(x, func, epsilon=1e-8)
    return x, converge_vals


func = rosen
x = np.array([1.2, 0.5])
print(func(x))
x_fin, vals = grad_descent(rosen, x)
print(func(x_fin))

plt.plot(np.log(vals))
plt.show()