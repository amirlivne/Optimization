from DNN import *
from BFGS import BFGS
import matplotlib.pyplot as plt


def eval_func(x):
    return x[:, 0] * np.exp(-np.sum(x**2, axis=1))


def make_DNN():
    dnn = DNN()
    dnn.set_input_size(2)
    dnn.add_layer(4, tanh_activation)
    dnn.add_layer(3, tanh_activation)
    dnn.add_layer()
    dnn.compile(loss=mse_loss, optimizer=BFGS)
    return dnn


# randomly choose test and train sets
np.random.seed()
n_train = 500
x_train = np.random.uniform(-2, 2, size=(n_train, 2))
y_train = eval_func(x_train)

n_test = 200
x_test = np.random.uniform(-2, 2, size=(n_test, 2))
y_test = eval_func(x_test)

# train the DNN
dnn = make_DNN()
m = dnn.train(x_train, y_train)

# plot results
print(len(m))
plt.semilogy(m)
plt.title('train loss (epoch)')
plt.ylabel('log(loss)')
plt.xlabel('epochs')
plt.savefig('./convergence graph.png')
plt.show()

# evaluate on test set
test_loss = dnn.evaluate(x_test, y_test)
print("test_loss: ", test_loss)
