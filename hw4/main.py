from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from AugmentedLagrangian import AugmentedLagrangianSolver


def PlotArea():
    # plot 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X1 = np.arange(-3, 1.5, 0.0005)
    X2 = np.arange(-1.5, 3, 0.0005)

    X1, X2 = np.meshgrid(X1, X2)
    Z = 2*(X1-5)**2 + (X2-1)**2

    X1_g = np.array([x1 for x1, x2 in zip(X1.flatten(), X2.flatten()) if x2>=abs(x1) and x2<=(1-x1/2)])
    X2_g = np.array([x2 for x1, x2 in zip(X1.flatten(), X2.flatten()) if x2>=abs(x1) and x2<=(1-x1/2)])
    Z_g = 2*(X1_g-5)**2 + (X2_g-1)**2

    print('min value: ', np.min(Z_g)*3)
    print('min value at point: ', X1_g[Z_g.argmin()], X2_g[Z_g.argmin()])

    # Plot the surface.
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot(X1_g, X2_g, zs=Z_g)
    ax.set_xlabel('X1'), ax.set_ylabel('X2'), ax.set_title('3D projection of feasible area')
    plt.show()

    # plot 2D
    X1 = np.arange(-3, 1.5, 0.0001)

    g1 = 1-X1/2
    g2 = X1
    g3 = -X1
    C_list = [30, 40, 50, 60, 70, 80, 90, 100, 110]
    plt.plot(g1, 'r')
    plt.plot(g2, 'b')
    plt.plot(g3, 'g')
    for c in C_list:
        level1_p = 1 + np.sqrt(c - 2 * (X1 - 5) ** 2)
        level1_m = 1 - np.sqrt(c - 2 * (X1 - 5) ** 2)
        plt.plot(level1_p, 'g', linewidth=0.5)
        plt.plot(level1_m, 'g', linewidth=0.5)
    plt.xticks(np.arange(0, len(X1), 5000), np.arange(-3, 1.5, 0.5))
    plt.legend(['g1', 'g2', 'g3'])
    plt.grid(color='r', linestyle='-', linewidth=0.2)
    plt.title('2D projection of feasible area')
    plt.show()


def test_func(x, nargout=1):
    val = 2*(x[0]-5)**2 + (x[1]-1)**2
    g = np.array([4*(x[0]-5), 2*(x[1]-1)])
    H = np.array([[4, 0], [0, 2]], dtype=np.float64)
    if nargout == 3:
        return val, g, H
    if nargout == 2:
        return val, g
    if nargout == 1:
        return val


def g1(x, nargout=1):
    val = x[1] + x[0]/2 - 1
    g = np.array([0.5, 1.])
    H = np.zeros((2, 2), dtype=np.float64)
    if nargout == 3:
        return val, g, H
    if nargout == 2:
        return val, g
    if nargout == 1:
        return val


def g2(x, nargout=1):
    val = x[0] - x[1]
    g = np.array([1., -1.])
    H = np.zeros((2, 2), dtype=np.float64)
    if nargout == 3:
        return val, g, H
    if nargout == 2:
        return val, g
    if nargout == 1:
        return val


def g3(x, nargout=1):
    val = - x[0] - x[1]
    g = np.array([-1, -1])
    H = np.zeros((2, 2), dtype=np.float64)
    if nargout == 3:
        return val, g, H
    if nargout == 2:
        return val, g
    if nargout == 1:
        return val


def DefineQuadFunc(Q, e, d):
    def QuadFunc(x, nargout=1):
        f = 0.5 * x.T @ Q @ x + d.T @ x + e
        if nargout == 1:
            return f
        g = 0.5 * (Q + Q.T) @ x + d
        if nargout == 2:
            return f, g
        if nargout == 3:
            H = 0.5 * (Q.T + Q)
            return f, g, H
    return QuadFunc


def DefineConstrains(A, b):
    def Constrains(x, nargout=1):
        f = A @ x + b
        g = A
        H = 0
        if nargout == 1:
            return f
        if nargout == 2:
            return f, g
        if nargout == 3:
            return f, g, H
    return Constrains


# true values
x = np.array([2/3, 2/3])
val = test_func(x)

# optimize
constrains = [g1, g2, g3]
lambda_vals = np.array([12, 34/3, 0])
xk, fx_list, xk_list, lambda_list, gradients = AugmentedLagrangianSolver(x_init=2*np.ones(2),
                                                                  func=test_func,
                                                                  constrains=constrains)

# plot results
plt.subplot(221)
f_x_diff = np.abs(fx_list-val)
plt.semilogy(f_x_diff)
plt.title('|f(x) - f(x*)| - log scale')
plt.xlabel('global Newton iterations')

plt.subplot(222)
x_diff = np.array([np.linalg.norm(xk - x) for xk in xk_list])
lambda_diff = np.array([np.linalg.norm(lambda_vals - l) for l in lambda_list])
plt.semilogy(x_diff, 'b')
plt.semilogy(lambda_diff, 'r')
plt.title('|x - x*| - log scale'), \
plt.legend(['|x - x*|', '|$\lambda - \lambda$*|'])
plt.xlabel('global Newton iterations')

plt.subplot(223)
g_violation = np.array([np.max([np.maximum(g(x), 0) for g in constrains]) for x in xk_list])
plt.semilogy(g_violation)
plt.xlabel('global Newton iterations')
plt.title('Max constrains violation - log scale')

plt.subplot(224)
plt.semilogy(gradients)
plt.title('|Grad(F(x))| - log scale')
plt.xlabel('global Newton iterations')
plt.show()


Q = np.array([[4., 0],
              [0, 2.]])
e = np.array([51.])
d = np.array([-21., -2.])
quad_func = DefineQuadFunc(Q, d, e)

A = np.array([[0.5, 1],
              [1, -1],
              [-1, -1]])
b = np.array([1., 0, 0])
constrains = DefineConstrains(A, b)

