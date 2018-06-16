from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from AugmentedLagrangian import AugmentedLagrangianSolver


def PlotArea():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X1 = np.arange(-3, 1.5, 0.05)
    X2 = np.arange(-1, 3, 0.05)

    X1, X2 = np.meshgrid(X1, X2)
    Z = 2*(X1-5)**2 + (X2-1)**2

    X1_g = np.array([x1 for x1, x2 in zip(X1.flatten(), X2.flatten()) if x2>=abs(x1) and x2<=(1-x1/2)])
    X2_g = np.array([x2 for x1, x2 in zip(X1.flatten(), X2.flatten()) if x2>=abs(x1) and x2<=(1-x1/2)])
    Z_g = 2*(X1_g-5)**2 + (X2_g-1)**2

    print('min value: ', np.min(Z_g))
    print('min value at point: ', X1_g[Z_g.argmin()], X2_g[Z_g.argmin()])

    # Plot the surface.
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot(X1_g, X2_g, zs=Z_g)
    ax.set_xlabel('X1'), ax.set_ylabel('X2'), ax.set_title('3D projection of feasible area')
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


# true values
x = np.array([2/3, 2/3])
val = test_func(x)

# optimize
constrains = [g1, g2, g3]
xk, fx_list, xk_list, lambda_list, gradients = AugmentedLagrangianSolver(x_init=2*np.ones(2),
                                                                  func=test_func,
                                                                  constrains=constrains)

# plot results
plt.subplot(221)
f_x_diff = np.abs(fx_list-val)
f_x_diff[f_x_diff==0] = 1.e-20
plt.plot(np.log(f_x_diff))
plt.title('|f(x) - f(x*)| - log scale')

plt.subplot(222)
x_diff = np.array([np.linalg.norm(xk - x) for xk in xk_list])
x_diff[x_diff==0] = 1.e-20
plt.plot(np.log(x_diff))
plt.title('|x - x*| - log scale')

plt.subplot(223)
g_violation = np.array([np.max([np.maximum(g(x), 0) for g in constrains]) for x in xk_list])
g_violation[g_violation == 0] = 1.e-20
plt.plot(np.log(g_violation))
plt.title('Max constrains violation - log scale')

plt.subplot(224)
gradients = np.array(gradients)
gradients[gradients == 0] = 1.e-20
plt.plot(np.log(gradients))
plt.title('|Grad(F(xk))| - log scale')
plt.show()

