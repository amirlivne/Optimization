# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:58:32 2018

@author: carmelr
"""

import numpy as np
from scipy.optimize import approx_fprime


# phi = sinx(x1*x2*x3)
def phi(x, nargout=1):
    value = np.sin(np.prod(x))
    if nargout == 3:
        grad = np.cos(np.prod(x)) * np.array([x[1] * x[2], x[0] * x[2], x[1] * x[0]])
        hessian = -value * (np.prod(x) ** 2 / np.outer(x, x)) + np.cos(np.prod(x)) * np.array(
            [[0, x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])
        return value, grad, hessian
    else:
        return value


# h = exp(x)
def h(x, nargout=1):
    if nargout == 3:
        return np.exp(x), np.exp(x), np.exp(x)
    else:
        return np.exp(x)


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

    if nargout == 3:
        value, grad, hessian = func(np.dot(A, x), nargout)
        return value, np.dot(grad, A), np.dot(np.transpose(A), hessian).dot(A)
        # return value, np.matmul(np.transpose(A), grad), np.matmul(np.transpose(A), hessian).__matmul__(A)

    else:
        return func(np.dot(A, x))


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

    if nargout == 3:
        in_value, in_grad, in_hessian = inner_func(x, **kwargs)
        value, first_d, sec_d = scalar_func(in_value, nargout=3)
        grad = first_d * in_grad
        hessian = sec_d * np.outer(in_grad, in_grad) + in_hessian * first_d
        return value, grad, hessian
    else:
        return scalar_func(inner_func(x, **kwargs))


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

# eps = np.sqrt(np.finfo(float).eps)
# result = approx_fprime(a, phi, eps)
#
# value, grad, hessian2 = f1(a, nargout=3, A=A)
# hessian_ = hessian(a, f1)
