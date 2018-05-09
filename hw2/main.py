import matplotlib.pyplot as plt
from hw2.functions import *
from hw2.GradDescent import *


if __name__ == "__main__":

    # Rosenbrok
    func = rosenbrok
    x_init = np.array([-1., -1.])
    x_opt = np.ones(2)
    f_opt = func(x_opt)
    x_fin, vals = grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, Rosenbrok')
    plt.show()


    # well-conditioned quadratic function
    func = quad_func
    x_init = np.array([2., -5.1, 5.])
    x_opt = np.zeros(3)
    f_opt = func(x_opt)
    x_fin, vals = grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, well-conditioned quadratic function')
    plt.show()


    # Rosenbrok newton
    func = rosenbrok
    x_init = np.array([-1., -1.])
    x_opt = np.ones(2)
    f_opt = func(x_opt)
    x_fin, vals = newton_grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, Rosenbrok with newton method')
    plt.show()

    # well-conditioned quadratic function, newton
    func = quad_func
    x_init = np.array([2., -5.1, 5.])
    x_opt = np.zeros(3)
    f_opt = func(x_opt)
    x_fin, vals = newton_grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, well-conditioned quadratic function, newton')
    plt.show()
