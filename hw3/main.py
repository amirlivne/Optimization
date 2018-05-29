from DNN import *

dnn = DNN()
dnn.set_input_size(2)
dnn.add_layer(4, tanh_activation)
dnn.add_layer(3, tanh_activation)
dnn.add_layer()
dnn.compile(loss=mse_loss, optimizer=BFGS)
# w = dnn.get_weights()
x = np.random.randn(10, 2)
y = eval_func(x)
# y_pred = dnn.forward(x)
# dnn.calc_grads(y_pred, y)
# weights = dnn.get_weights()
# dnn.x = x
# dnn.y = y
# value, grads = dnn.target_func(weights)
m = dnn.train(x, y)
print(m)

