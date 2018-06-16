import numpy as np
from GradDescent import newton_grad_descent
import copy

def QuadraticLogarithmicPenalty(x):

    if x >= -0.5:
        return x + (x**2)/2, x+1, 1
    else:
        return -3/8 - 0.25*np.log(-2*x), -0.25/x, 0.25/x**2


def Penalty_p_mu(x, p=1, mu=1, panelty_func=QuadraticLogarithmicPenalty, nargout=1):
    val, fisrt_d, seceond_d = panelty_func(p*x)
    if nargout == 3:
        return mu*val/p, mu*fisrt_d, mu*seceond_d*p
    if nargout == 2:
        return mu*val/p, mu*fisrt_d
    if nargout == 1:
        return mu*val/p


def AugmentedLagrangianSolver(x_init, func, constrains, pmax=1000, alfa=4, eps=1.e-5):
    """
    Augmented Lagrangian Solver
    :param x_init: numpy array of initial x0 point
    :param func: function to optimize
    :param constrains: list of functions g(x) represents inequities constrains g(x) <= 0
    :param pmax: maximum value of p factor
    :param alfa: alfa factor for updating p factor
    :return: optimized x point and list of convergence func(x[k])
    """

    # init
    lambda_vec = np.ones(len(constrains))
    xk = x_init
    pk = 2

    # init results lists
    fx_list = [func(x_init)]
    xk_list = [copy.deepcopy(x_init)]
    lambda_list = [copy.deepcopy(lambda_vec)]
    gradients = []

    # iterate
    while True:

        # define updated F penalty function
        def Fpenalty(x, nargout=1):
            val, g, H = func(x, nargout=3)
            for i, g_func in enumerate(constrains):
                val_i, g_i, H_i = g_func(x, nargout=3)
                val_p, d1_p, d2_p = Penalty_p_mu(val_i, p=pk, mu=lambda_vec[i], nargout=3)
                val += val_p
                g += g_i * d1_p
                H += H_i * d1_p + d2_p * np.outer(g_i, g_i)
            if nargout == 3:
                return val, g, H
            if nargout == 2:
                return val, g
            if nargout == 1:
                return val

        # stop condition
        _, grad = Fpenalty(xk, nargout=2)
        norm = np.linalg.norm(grad)
        gradients.append(norm)
        if norm < eps:
            break

        # optimize the F penalty function using newton method
        xk, t = newton_grad_descent(Fpenalty, xk)
        # plt.plot(t)
        # plt.show()

        # update all lambda[k] values
        for i, g in enumerate(constrains):
            _, g_val = Penalty_p_mu(g(xk), p=pk, mu=1, nargout=2)
            lambda_vec[i] *= np.maximum(np.minimum(g_val, 3), 1/3)

        # update p[k]
        pk = np.minimum(alfa*pk, pmax)

        # save results
        fx_list.append(func(xk))
        xk_list.append(copy.deepcopy(xk))
        lambda_list.append(copy.deepcopy(lambda_vec))

    return xk, fx_list, xk_list, lambda_list, gradients


