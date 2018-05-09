from hw2.mcholmz import modifiedChol
import numpy as np

def armijo(x, func, d, sigma=0.25, beta=0.5):
    """
    Armijo line search
    :param x: point
    :param func: a function to optimize
    :param d: direction of line for the line search
    :param sigma: parameter for stop condition
    :param beta: divide factor to find optimal alfa
    :return: alfa as float - the optimal step size
    """

    # init alfa
    alfa = 1.

    # compute c (the diffrencial of phi(alfa) at alfa=0)
    # grad_x = approx_fprime(x, func, epsilon=1e-8)
    _, grad_x = func(x, nargout=2)
    c = np.dot(grad_x, d)

    while (func(x + alfa*d) - func(x)) > sigma*c*alfa:
        alfa *= beta

    return alfa


def grad_descent(func, x_init, eps=10e-5):
    """
    conpute gradien descent
    :param func: a function to optimize
    :param x_init: initial point
    :param eps: for stop conditions
    :return: optimal point x, and list of f(x) values during optimization
    """
    x = x_init
    converge_vals = []
    while True:
        converge_vals.append(func(x))
        # grad_x = approx_fprime(x, func, epsilon=1e-8)
        _, grad_x = func(x, nargout=2)
        d = -grad_x
        x += armijo(x, func, d)*d
        if np.linalg.norm(grad_x) <= eps:
            break
    return x, converge_vals


def newton_grad_descent(func, x_init, eps=10e-5):
    """
    conpute gradien descent using newton method
    :param func: a function to optimize
    :param x_init: initial point
    :param eps: for stop conditions
    :return: optimal point x, and list of f(x) values during optimization
    """
    x = x_init
    converge_vals = []
    while True:
        converge_vals.append(func(x))
        # grad_x = approx_fprime(x, func, epsilon=1e-8)
        _, grad_x, H = func(x, nargout=3)
        L, D, e = modifiedChol(H)
        # Err = (L @ np.diag(D.flatten()) @ L.T) - H
        d = solve_direction(L, D, grad_x)
        x += armijo(x, func, d)*d
        if np.linalg.norm(grad_x) <= eps:
            break
    return x, converge_vals


def solve_direction(L, D, g):
    n = len(g)

    # compute y=DLd from Ly=-g
    y = np.zeros(n)
    y[0] = -g[0]/L[0, 0]
    for i in range(1, n):
        y[i] = -(g[i] + np.dot(y[:i], L[i, :i]))/L[i, i]

    # compute z=y/D
    z = y/D[:,0]

    # compute d from Lt d = Z
    Lt = np.transpose(L)
    d = np.zeros(n)
    d[-1] = z[-1]/Lt[-1, -1]
    for i in np.arange(n-2, -1, -1):
        d[i] = (z[i] - np.dot(z[i+1:], Lt[i, i+1:]))/Lt[i, i]

    return d





