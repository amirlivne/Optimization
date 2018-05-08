import numpy as np
from scipy.optimize import approx_fprime, rosen


# f_x = rosen(x)
def rosenbrok_func(x):
    if len(x) == 1:
        print('error, vector must be at lease with 2 variables')
        return None
    else:
        return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def armijo(x, func, d, sigma=0.25, beta=0.5, max_iter=20):
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
    print(c, "c")
    print(func(x + alfa*d))
    print(sigma*c*alfa)
    print('\n')
    for i in range(max_iter):
        if func(x + alfa*d) > sigma*c*alfa:
            alfa *= beta
        else:
            break
        print(alfa)
        print(func(x + alfa * d))
        print(sigma * c * alfa)
        print('\n')

    return alfa


def grad_descent(func, x_init, eps=10e-5, max_iter=400):
    x = x_init
    grad_x = approx_fprime(x, func, epsilon=1e-8)
    for i in range(max_iter):
        if np.linalg.norm(grad_x) < eps:
            break
        else:
            x += x + armijo(x, func, -grad_x)
            grad_x = approx_fprime(x, func, epsilon=1e-8)

    return x


x = np.array([-1., -1.])
grad_x = approx_fprime(x, rosen, epsilon=1e-8)
func = rosen
alfa = armijo(x, func, -grad_x)


e= np.sqrt(np.finfo(float).eps)

x_fin = grad_descent(rosen, x)
print(func(x))

