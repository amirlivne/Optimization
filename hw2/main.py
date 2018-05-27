import matplotlib.pyplot as plt
from functions import *
from GradDescent import *

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
    func = quad_func_well
    x_init = np.ones(10)
    x_opt = np.zeros(10)
    f_opt = func(x_opt)
    x_fin, vals = grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] = 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, well-conditioned quadratic function')
    plt.show()

    # ill-conditioned quadratic function
    func = quad_func_ill
    x_init = np.ones(10)
    x_opt = np.zeros(10)
    f_opt = func(x_opt)
    x_fin, vals = grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20  # for legal log values

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, ill-conditioned quadratic function')
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
    func = quad_func_well
    x_init = np.ones(10)
    x_opt = np.zeros(10)
    f_opt = func(x_opt)
    x_fin, vals = newton_grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, well-conditioned quadratic function, newton')
    plt.show()

    # well-conditioned quadratic function, newton
    func = quad_func_ill
    x_init = np.ones(10)
    x_opt = np.zeros(10)
    f_opt = func(x_opt)
    x_fin, vals = newton_grad_descent(func, x_init)

    vals -= f_opt
    vals[vals == 0.] += 1.e-20

    plt.figure()
    plt.plot(np.log(vals)), plt.title('f(x) in log scale, ill-conditioned quadratic function, newton')
    plt.show()
