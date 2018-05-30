from DNN import *
from BFGS import BFGS
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import approx_fprime


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

# define the model
dnn = make_DNN()

# test the gradient
weights = dnn.get_weights()
dnn.x = x_train[0]
dnn.y = y_train[0]
print("x: ", x_train[0])
print("y: ", y_train[0])
print("weights vec: ", weights)
_, analytical_grads = dnn.target_func(weights, nargout=2)
approx_grads = approx_fprime(weights, dnn.target_func, 1e-8)
diff = analytical_grads - approx_grads
print("the mean diff is: ", np.mean(diff))
print("the sum diff is: ", np.sum(diff))
print("the std diff is: ", np.std(diff))

# train the DNN
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
y_pred = dnn.forward(x_test)
test_loss = dnn.evaluate(x_test, y_test)
print("test_loss: ", test_loss)

# plot 3d graph of target function
a = np.arange(-2, 2, 0.2)
X1, X2 = np.meshgrid(a, a)
Y = X1 * np.exp(-X1 ** 2 - X2 ** 2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, alpha=0.5)
fig.colorbar(surf, shrink=0.5, aspect=5)
for i in range(n_test):
    ax.scatter(x_test[i, 0], x_test[i, 1], y_pred[i], c='g', marker='^')
plt.show()

