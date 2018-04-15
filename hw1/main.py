import numpy as np
from hw1.task3 import *
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    args = args[1]  # the args to pass to the function
    # calc grad and hessian
    output_grad = np.zeros((len(x), 1))
    output_hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        val_plus, g_plus, _ = myfunc(x + epsilon * e_mat[i], **args)
        val_minus, g_minus, _ = myfunc(x - epsilon * e_mat[i], **args)
        output_grad[i] = (val_plus - val_minus) / (2 * epsilon)
        output_hessian[:, i] = (g_plus - g_minus) / (2 * epsilon)
    return output_grad, output_hessian

### task 5
def analyze(f1_res, f2_res, numeric_list, input, epsilons):
    num_f1_res, num_f2_res = numeric_list[3]
    values, grads, hessians = [[x, y] for x, y in zip(f1_res, f2_res)]
    num_grads, num_hessians = [[x, y] for x, y in zip(num_f1_res, num_f2_res)]

    grad_per_eps, hes_per_eps = [[],[]],[[],[]]

    for (num1, num2) in numeric_list:
        grad_diff1 = num1[0].reshape(grads[0].shape) - grads[0]
        hess_diff1 = num1[1] - hessians[0]
        grad_diff2 = num2[0].reshape(grads[1].shape) - grads[1]
        hess_diff2 = num2[1] - hessians[1]
        grad_per_eps[0].append(np.amax(abs(grad_diff1)))
        grad_per_eps[1].append(np.amax(abs(grad_diff2)))
        hes_per_eps[0].append(np.amax(abs(hess_diff1)))
        hes_per_eps[1].append(np.amax(abs(hess_diff2)))

    for i in range(2):
        # init figure
        fontsize = 9
        fig = plt.figure(i + 1)
        fig.suptitle("f{} Analysis".format(i + 1), fontsize=14)
        ax1, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)
        grad_diff = num_grads[i].reshape(grads[0].shape) - grads[i]
        hess_diff = num_hessians[i] - hessians[i]
        # plot grad
        ax1.plot(input, grad_diff), ax1.set_title("Gradient Difference", fontsize=fontsize)
        # plot hessian
        ax2.imshow(hess_diff, extent=[0, 1, 0, 1]), ax2.set_title("Hessian Difference", fontsize=fontsize)
        # plot max grad per epsilon
        ax3.semilogx(epsilons, grad_per_eps[i]), ax3.set_title("Max abs diff of gradient per epsilon", fontsize=fontsize)
        # plot max grad per epsilon
        ax4.semilogx(epsilons, hes_per_eps[i]), ax4.set_title("Max abs diff of hess per epsilon", fontsize=fontsize)
        plt.show()


if __name__ == "__main__":
    # inputs = np.random.randint(10, size=3)
    inputs = np.array((1.5, 2.7, 3))
    # A = np.random.randn(3, 3)
    A = np.array(([1, 2, 3], [5, 2, 1], [1, 2, 7]))
    args1 = {'func':phi, 'nargout':3, 'A':A}
    args2 = {'nargout':3}
    values = np.linspace(0, 10, 1000)
    epsilons = [(2 ** -(16-x)) * max(abs(inputs)) for x in values]
    # print(np.amin(epsilons), np.amax(epsilons))
    f1_res = f1(inputs, **args1)
    f2_res = f2(inputs, **args2)
    numeric_list = []
    for eps in tqdm(epsilons):
        num_f1 = numdiff(f1, inputs, eps, args1)
        num_f2 = numdiff(f2, inputs, eps, args2)
        numeric_list.append((num_f1,num_f2))
    analyze(f1_res, f2_res, numeric_list, inputs, epsilons)
