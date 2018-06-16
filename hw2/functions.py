import numpy as np
from scipy.io import loadmat


def rosenbrok(x, nargout=1):
    n = len(x)
    f = sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    if nargout == 1:
        return f
    elif nargout > 3:
        print('error, illegal nargout value')
        return None
    g = np.zeros(n)
    g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    g[-1] = 200*(x[-1] - x[-2]**2)
    g[1:-1] = -2*(1 - x[1:-1]) - 400*x[1:-1]*(x[2:] - x[1:-1]**2) + 200*(x[1:-1] - x[0:-2]**2)
    if nargout == 2:
        return f, g
    else:   # nargout == 3
        H = np.diag(-400*x[:-1], 1) - np.diag(400*x[:-1], -1)
        diagonal = np.zeros((n, n))
        diagonal[0,0] = 1200*x[0]**2 - 400*x[1] + 2
        diagonal[-1, -1] = 200
        if n > 2:
            diagonal[1:-1, 1:-1] = np.diag(202 + 1200*x[1:-1]**2 - 400*x[2:])
        H += diagonal
        return f, g, H


def quad_func_well(x, nargout=1):
    Q = loadmat('h')['H1']
    print(Q)
    f = 0.5 * x.T @ Q @ x
    if nargout == 1:
        return f
    elif nargout > 3:
        print('error, illegal nargout value')
        return None
    g = 0.5*(Q+Q.T) @ x
    if nargout == 2:
        return f, g
    else:   # nargout == 3
        H = 0.5*(Q.T + Q)
        return f, g, H


def quad_func_ill(x, nargout=1):
    Q = loadmat('h')['H2']

    f = 0.5 * x.T @ Q @ x
    if nargout == 1:
        return f
    elif nargout > 3:
        print('error, illegal nargout value')
        return None
    g = 0.5*(Q+Q.T) @ x
    if nargout == 2:
        return f, g
    else:   # nargout == 3
        H = 0.5*(Q.T + Q)
        return f, g, H
