import numpy as np
import scipy.optimize

### Armijo inexact line search params ###
a0 = 1
sigma = 0.25
beta = 0.5
eps = 5.e-5


def ArmijoLineSearch(func, value, grad, x, d_k):
    """
    performing inexact line search, in order to find an acceptable step length that
    reduces the objective function 'sufficiently'.
    :param func: the function to reduce
    :param value: func value at current guess
    :param grad: func grad at current guess
    :param x: current guess
    :param d_k: current direction
    :return: a: best step size, or None in case of error (if can't ensure positive definiteness of the Hessian model
                after the update)
    """
    # initilize first iteration:
    a = a0
    new_value = func(x+a*d_k)
    y_k = new_value - value
    c = np.matmul(grad.T, d_k)
    # stop_limit = sigma * a * c

    # loop until Armijo condition is satisfied:
    while y_k > sigma * a * c:
        a = beta*a
        new_value = func(x + a * d_k)
        y_k = new_value - value

    # check that the directional derivative at current step length a is less negative
    # than for a = 0
    _, new_grad = func(x + a * d_k, nargout=2)
    c_new = np.matmul(new_grad.T, d_k)
    if c_new >= c:
        return a, True
    else:
        return a, False


def BFGS(func, x0):
    """
    calculates the min of "func", by starting from x0 point
    :param func: the target function to minimize.
                 accepts a vector x and returns a real scalar, the objective function
                 evaluated at x and the gradient of the objective function at this point.
    :param x0: the starting point. a row numpy array, shaped (len,?), for example np.array([3,4,5])
    :return: m: a vector with the values of the target function fun x ( ) k at iteration k
             x: the solution (point of min), returned as a real vector. The size of x is the same as the size of x0 . x is a
                local solution to the problem
    """

    ### Initial guess of hessian matrix is I ###
    B = np.identity(len(x0))
    x = x0.reshape(len(x0), 1)
    val, g = func(x0, nargout=2)
    g = g.reshape(len(g),1)
    m = [val]
    while np.linalg.norm(g) > eps:
        # find direction according to Newton:
        d_k = -np.matmul(B, g)
        # find step size using Armijo inexact line search:
        a, a_is_valid = ArmijoLineSearch(func, val, g, x, d_k)
        # update results:
        x_new = x + a * d_k
        val, g_new = func(x_new.reshape(len(x_new)), nargout=2)
        g_new = g_new.reshape(len(g_new), 1)
        m.append(val)
        if a_is_valid:
            if np.matmul(g.T, d_k) < np.matmul(g_new.T, d_k):
                p = x_new - x
                q = g_new - g
                s = np.matmul(B, q)
                tau = np.matmul(s.T, q)
                mu = np.matmul(p.T, q)
                v = p/mu - s/tau
                # update hessian:
                B = B + np.matmul(p, p.T) / mu - np.matmul(s, s.T) / tau + np.matmul(v, v.T) * tau
            else:
                break
        else: # meaning can't ensure positive hessian. don't update hessian!
            pass
        # update func value and grad
        g = g_new
        x = x_new

    return x, m


def rosenbrock(X, nargout=1):
    val = scipy.optimize.rosen(X)
    grad = scipy.optimize.rosen_der(X)
    if nargout == 1:
        return val
    return val, grad


def test_func(X, nargout=1):
    val = X[0]*np.sum(np.exp(-np.sum(X**2)))
    dx = (1-2*X[0]**2)*np.sum(np.exp(-np.sum(X**2)))
    dy = -2*X[0]*X[1]*np.sum(np.exp(-np.sum(X**2)))
    if nargout == 1:
        return val
    return val, np.array([dx, dy])


if __name__ == '__main__':
    m, min_x = BFGS(rosenbrock, np.array([0.25, 4, 10]))
    # m, min_x = BFGS(test_func, np.array([-2, 0.5]))
    print("min is at", min_x)
    print(m)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot([i for i in range(len(m))], m)
    # plt.title("Rosenbrock function minimization using BFGS (N=8)")
    # plt.xlabel("Iteration")
    # plt.ylabel("Rosenbrock Value")
    # plt.show()

