import numpy as np
import scipy.optimize

### Armijo inexact line search params ###
a0 = 1
sigma = 0.25
beta = 0.5
eps = 10 ** -5


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
    new_value, _ = func(x+a*d_k)
    y_k = new_value - value
    c = np.matmul(d_k, np.transpose(grad))
    # stop_limit = sigma * a * c

    # loop until Armijo condition is satisfied:
    while y_k > sigma * a * c:
        a = beta*a
        new_value, _ = func(x + a * d_k)
        y_k = new_value - value

    # check that the directional derivative at current step length a is less negative
    # than for a = 0
    _, new_grad = func(x + a * d_k)
    c_new = np.matmul(d_k, np.transpose(new_grad))
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
    x = x0
    val, g = func(x)
    m = [val]

    while np.linalg.norm(g) > eps:
        # find direction according to Newton:
        d_k = -np.matmul(B, np.transpose(g))
        # find step size using Armijo inexact line search:
        a, a_is_valid = ArmijoLineSearch(func, val, g, x, d_k.T)
        # update results:
        x_new = x + a * d_k.T
        val, g_new = func(x_new)
        m.append(val)
        if a_is_valid:
            p = x_new.reshape(1,len(x)) - x.reshape(1,len(x))
            q = g_new.reshape(1,len(x)) - g.reshape(1,len(x))
            s = np.matmul(B, q.T).T
            tau = np.matmul(s, q.T)
            mu = np.matmul(p, q.T)
            v = p/mu - s/tau
            # update hessian:
            B = B + np.matmul(p.T, p) / mu - np.matmul(s.T, s) / tau + np.matmul(v.T, v) * tau
        else: # meaning can't ensure positive hessian. don't update hessian!
            pass
        # update func value and grad
        g = g_new
        x = x_new

    return m, x


def rosenbrock(X):
    val = scipy.optimize.rosen(X)
    grad = scipy.optimize.rosen_der(X)
    return val, grad


if __name__ == '__main__':
    m, min_x = BFGS(rosenbrock, np.array([0.25, 4, 10, 5, 10, -2, 0.023, 5]))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([i for i in range(len(m))], m)
    plt.title("Rosenbrock function minimization using BFGS (N=8)")
    plt.xlabel("Iteration")
    plt.ylabel("Rosenbrock Value")
    plt.show()

