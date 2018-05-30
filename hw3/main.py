from DNN import *
from BFGS import BFGS


def make_DNN():
    dnn = DNN()
    dnn.set_input_size(2)
    dnn.add_layer(4, tanh_activation)
    dnn.add_layer(3, tanh_activation)
    dnn.add_layer()
    dnn.compile(loss=mse_loss, optimizer=BFGS)
    return dnn


np.random.seed(0)
dnn = make_DNN()
x = np.random.randn(400, 2)
y = eval_func(x)
# x = np.ones((1, 2))
# y = np.ones(1)
# w = dnn.get_weights()
# dnn.set_weights(w)
# w2 = dnn.get_weights()
# dnn.x = x
# dnn.y = y
# y_pred = dnn.forward(x)
# loss, grad = dnn.loss(y_pred, y, nargout=2)
# dnn.calc_grads(y_pred, y)
# w3 = dnn.get_weights()
# grads = dnn.get_grads()
# # w = dnn.get_weights()
# # val, grad = dnn.target_func(w)
# # w1 = 1 + w
# # val2, grad2 = dnn.target_func(w1)
# # w2 = dnn.get_weights()
# y_pred = dnn.forward(x)
# dnn.calc_grads(y_pred, y)
# dnn.get_grads()
# weights = dnn.get_weights()
# dnn.x = x
# dnn.y = y
# value, grads = dnn.target_func(weights)
m = dnn.train(x, y)
print(m)

