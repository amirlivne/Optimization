import numpy as np
from DNN import *
from optimizers import SGD, AdaGrad, BFGS
import matplotlib.pyplot as plt


def eval_func(x):
    return x[:, 0] * np.exp(-np.sum(x**2, axis=1))


def make_DNN(lr=0.1, exp_decay=0.001, optimizer='SGD'):
    dnn = DNN()
    dnn.set_input_size(2)
    dnn.add_layer(4, tanh_activation)
    dnn.add_layer(3, tanh_activation)
    dnn.add_layer(1, linear_activation)
    if optimizer == 'SGD':
        dnn.compile(loss=mse_loss, optimizer=SGD(lr=lr, exp_decay=exp_decay))
    elif optimizer == 'Ada':
        dnn.compile(loss=mse_loss, optimizer=AdaGrad(lr=lr, exp_decay=exp_decay))
    elif optimizer == 'BFGS':
        dnn.compile(loss=mse_loss, optimizer=BFGS)
    return dnn


def hyper_lr(x_train, y_train, x_test, y_test, optimizer='SGD', experiments=10):

    results = np.zeros((experiments, 3))
    for i in range(experiments):

        # randomly select lr and decay
        lr = 10**np.random.uniform(-0.05, -2)
        exp_decay = 10**np.random.uniform(-3, -8)
        # lr = 10**np.random.uniform(0.05, -3)
        # exp_decay = 10**np.random.uniform(-2, -6)

        # build and train net
        dnn = make_DNN(lr=lr, exp_decay=exp_decay, optimizer=optimizer)
        dnn.train(x_train, y_train, epochs=1000, shuffle=False)

        # save results
        test_loss = dnn.evaluate(x_test, y_test)
        results[i, :] = np.array([test_loss, lr, exp_decay])
        print('finished experiment ', i)

    # sort by loss
    index = np.argsort(results[:, 0], axis=0)
    results = results[index]
    return results


def hyper_batch_size(x_train, y_train, x_test, y_test, lr, exp_decay, optimizer='SGD'):

    mini_batch_list = np.concatenate(([1], np.arange(10, 501, 10, dtype=int)))
    # mini_batch_list = np.arange(1, 11, 1, dtype=int)
    results = np.zeros((len(mini_batch_list), 2))

    # build and train net, always use same initializer
    for i, batch_size in enumerate(mini_batch_list):
        dnn = make_DNN(lr=lr, exp_decay=exp_decay, optimizer=optimizer)
        dnn.train(x_train, y_train, epochs=1000, shuffle=False, batch_size=batch_size)

        # save results
        test_loss = dnn.evaluate(x_test, y_test)
        results[i, :] = np.array([test_loss, batch_size])
        print('finished experiment ', i)

    return results, mini_batch_list


# randomly choose test and train sets
np.random.seed(1)
n_train = 500
x_train = np.random.uniform(-2, 2, size=(n_train, 2))
y_train = eval_func(x_train)

n_test = 200
x_test = np.random.uniform(-2, 2, size=(n_test, 2))
y_test = eval_func(x_test)


#### TASK 2 ####
optimizer = 'SGD'

# section 1 - run lr hyper parameter
lr_results = hyper_lr(x_train, y_train, x_test, y_test, experiments=100, optimizer='SGD')
# lr_results = hyper_lr(x_train, y_train, x_test, y_test, experiments=100, optimizer='Ada')
print(lr_results[:5, :])
np.savetxt('./hyper_lr_{}.txt'.format(optimizer), lr_results)
[best_lr, best_exp_decay] = lr_results[0, 1:]


# section 2 - run batch size hyper parameter
batch_size_results, mini_batch_list = hyper_batch_size(x_train, y_train, x_test, y_test, lr=0.5, exp_decay=1.*10**-4, optimizer='SGD')
# batch_size_results, mini_batch_list = hyper_batch_size(x_train, y_train, x_test, y_test, lr=0.2, exp_decay=5.*10**-5, optimizer='Ada')
print(batch_size_results)
np.savetxt('./hyper_batch_size_{}.txt'.format(optimizer), batch_size_results)
best_batch_size = batch_size_results[0, 1]

# batch_size_results = np.loadtxt('./hyper_batch_size.txt')
plt.semilogy(batch_size_results[:, 1], batch_size_results[:, 0])
plt.title('test loss (mini-batch size)', fontsize=20)
plt.ylabel('test loss', fontsize=20)
plt.xlabel('mini-batch size', fontsize=20)
plt.savefig('./test loss mini-batch size.png')
plt.show()


# section 3
# define and train the model
dnn = make_DNN(lr=0.5, exp_decay=10**-4, optimizer='SGD')
values, test_values = dnn.train(x_train, y_train, test=(x_test, y_test), epochs=5000, shuffle=False, batch_size=100)

# define and train the model
dnn = make_DNN(lr=0.5, exp_decay=10**-4, optimizer='SGD')
values_shuffle, sh_test_values = dnn.train(x_train, y_train, test=(x_test, y_test), epochs=5000, shuffle=True, batch_size=100)

# plot results
plt.semilogy(values, label='without shuffle - train')
plt.semilogy(values_shuffle, label='shuffle - train')
plt.semilogy(test_values, label='without shuffle - test')
plt.semilogy(sh_test_values, label='shuffle - test')
plt.title('loss (epoch)', fontsize=16)
plt.ylabel('loss', fontsize=16), plt.xlabel('epochs', fontsize=16), plt.legend()
plt.savefig('./section 3 graph.png')
plt.show()


#### TASK 3 ####
# define and train the model
dnn = make_DNN(lr=0.5, exp_decay=0.0001, optimizer='SGD')
values_SGD = dnn.train(x_train, y_train, epochs=1000, batch_size=10)

# define and train the model
dnn = make_DNN(lr=0.2, exp_decay=5.*10**-5, optimizer='Ada')
values_Ada = dnn.train(x_train, y_train, epochs=1000, batch_size=10)

# define and train the model
dnn = make_DNN(lr=0.2, exp_decay=5.*10**-5, optimizer='BFGS')
values_BFGS = dnn.train_bfgs(x_train, y_train, epochs=1000)

# plot results
plt.semilogy(values_SGD, label='SGD')
plt.semilogy(values_Ada, label='AdaGrad')
plt.semilogy(values_BFGS, label='BFGS')
plt.title('train loss (epoch)', fontsize=16)
plt.ylabel('loss - log scale', fontsize=16), plt.xlabel('epochs', fontsize=16), plt.legend()
plt.savefig('./task 3 graph.png')
plt.show()

