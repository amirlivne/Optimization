# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:58:32 2018

@authors: carmelr and amirlivne
"""

import numpy as np
from scipy.optimize import approx_fprime

#task 3
# phi = sinx(x1*x2*x3)
def phi(x, nargout=1):
    value = np.sin(np.prod(x))
    if nargout == 1:
        return value, None, None
    elif nargout == 2:
        grad = np.cos(np.prod(x)) * np.array([x[1] * x[2], x[0] * x[2], x[1] * x[0]])
        return value, grad, None
    elif nargout == 3:
        grad = np.cos(np.prod(x)) * np.array([x[1] * x[2], x[0] * x[2], x[1] * x[0]])
        hessian = -value * (np.prod(x) ** 2 / np.outer(x, x)) + np.cos(np.prod(x)) * np.array(
            [[0, x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])
        return value, grad, hessian


# h = exp(x)
def h(x, nargout=1):
    if nargout == 3:
        return np.exp(x), np.exp(x), np.exp(x)
    elif nargout == 2:
        return np.exp(x), np.exp(x), None
    else:
        return np.exp(x), None, None


# compute the value, grad and hessian of func(Ax)
def f1(x, **kwargs):
    nargout = 1
    if 'nargout' in kwargs:
        nargout = kwargs['nargout']
    if 'A' in kwargs:
        A = kwargs['A']
    else:
        A = np.eye(len(x))
    if 'func' in kwargs:
        func = kwargs['func']
    else:
        func = phi
    value, grad, hessian = func(np.dot(A, x), nargout)
    if nargout == 3:
        return value, np.dot(grad, A), np.dot(np.transpose(A), hessian).dot(A)
        # return value, np.matmul(np.transpose(A), grad), np.matmul(np.transpose(A), hessian).__matmul__(A)
    elif nargout == 2:
        return value, np.dot(grad, A)
    else:
        return value


# compute the value, grad and hessian of scalar_func(inner_func(x))
def f2(x, **kwargs):
    nargout = 1
    if 'nargout' in kwargs:
        nargout = kwargs['nargout']
    if 'scalar_func' in kwargs:
        scalar_func = kwargs['scalar_func']
    else:
        scalar_func = h

    if 'inner_func' in kwargs:
        inner_func = kwargs['inner_func']
    else:
        inner_func = phi

    in_value, in_grad, in_hessian = inner_func(x, **kwargs)
    value, first_d, sec_d = scalar_func(in_value, nargout=nargout)
    if nargout == 3:
        grad = first_d * in_grad
        hessian = sec_d * np.outer(in_grad, in_grad) + in_hessian * first_d
        return value, grad, hessian
    elif nargout == 2:
        grad = first_d * in_grad
        return value, grad
    else:
        return value


# task 4
def numdiff(myfunc, x, *args):
    """
    :param myfunc: function handler, will contain functions from tasks 3, to
                    obtain the function value and analytical gradient if needed
    :param x: vector that represents the point on which we calculate the output
    :param *args: all additional params for the calculation for example epsilon
            (increment of x) for the following task, and for the functions
    :return: gnum: numerical estimation of gradient.
             Hnum: numerical estimation of hessian.
    """
    N = len(x)
    epsilon = args[0]
    e_mat = np.identity(N)
    args = args[1]  # the args to pass to the function

    # calc grad and hessian
    output_grad = np.zeros((N, 1))
    output_hessian = np.zeros((N, N))
    for i in range(N):
        val_plus, g_plus = myfunc(x + epsilon * e_mat[i], **args)
        val_minus, g_minus = myfunc(x - epsilon * e_mat[i], **args)
        output_grad[i] = (val_plus - val_minus) / (2 * epsilon)
        output_hessian[:, i] = (g_plus - g_minus) / (2 * epsilon)
    return output_grad, output_hessian


# Hessian Matrix Aproximation
# from https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
def hessian(x, the_func):
    N = x.shape[0]
    hessian = np.zeros((N, N))
    gd_0 = approx_fprime(x, the_func, np.sqrt(np.finfo(float).eps))
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps
    for i in range(N):
        xx0 = 1. * x[i]
        x[i] = xx0 + eps
        gd_1 = approx_fprime(x, the_func, np.sqrt(np.finfo(float).eps))
        hessian[:, i] = ((gd_1 - gd_0) / eps).reshape(x.shape[0])
        x[i] = xx0
    return hessian


# runing example
# a = np.array([1, 2.3, 1.3])
# A = np.ones((3, 3)) - np.eye(3)

# f1(a, phi, nargout=3, A=A, func=phi)
# f2(a, scalar_func=h, inner_func=phi)
