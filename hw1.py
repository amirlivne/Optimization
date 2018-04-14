import numpy as np
from pika import *

### task 4
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
    epsilon = args[0]
    e_mat = np.identity(len(x))
    args = args[1:]
    # calc grad and hessian
    output_grad = np.zeros((len(x), 1))
    output_hessian = np.zeros((len(x), len(x)))
    print('inited values')
    for i in range(len(x)):
        val_plus, g_plus, _ = myfunc(x + epsilon * e_mat[i], *args)
        val_minus, g_minus, _ = myfunc(x - epsilon * e_mat[i], *args)
        output_grad[i] = (val_plus - val_minus)/(2*epsilon)
        output_hessian[:, i] = (g_plus - g_minus)/(2*epsilon)
    print('finished calc')
    return output_grad, output_hessian

if __name__ == "__main__":
    x = np.array((1,2,3))
    args1 = [phi, np.random.randn(3, 3)]
    args2 = [h, phi]
    epsilon = ((2 ** -16) ** (1 / 3)) * max(abs(x))

    # grad, hes = numdiff(f1, x, epsilon, *args1)
    print(f1(x, phi, np.random.randn(3, 3)))