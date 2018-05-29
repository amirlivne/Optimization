import numpy as np
import scipy.optimize.fmin_l_bfgs_b as bfgs


def loss_func(y_true, y_pred, nargout=1):

    loss = np.mean((y_pred - y_true)**2)
    if nargout == 2:
        grad = 2*np.mean(y_pred - y_true)
        return loss, grad
    else:
        return loss


def tanh_activation(v, nargout=1):
    val = np.tanh(v)
    if nargout == 2:
        grad = 1 - np.tanh(v)**2
        return val, grad
    else:
        return val


def linear_activation(v, nargout=1):
    val = v
    if nargout == 2:
        grad = 1
        return val, grad
    else:
        return val


class DNN:
    def __init__(self):
        self.layers = []
        self.input_size = 1
        self.optimizer = None
        self.loss = None
        self.curr_vals = None
        self.curr_grads = None
        self.batch_size = 1

    def set_input_size(self, size):
        self.input_size = size

    def add_layer(self, n=1, activation=linear_activation):
        if not self.layers:    # first layer
            m = self.input_size
        else:
            m = self.layers[-1]['w'].shape[1]
        w = np.random.randn(m, n) / np.sqrt(n)
        b = np.zeros(n)
        layer = {'w': w, 'w_length': m*n, 'b': b, 'b_length': n, 'activation': activation}
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.loss = loss
        self.optimizer = optimizer

    def get_weights(self):
        weights = np.empty(0)
        for layer in self.layers:
            weights = np.concatenate((weights, layer['w'].flatten(), layer['b']))
        return weights

    # def reshape_weights(self, weights):
    #     for layer
    #     shape =

    def target_func(self, weights):

        weights[:]

    def evaluate(self, x):  # x should be (input_size, batch_size)
        if x.ndim > 2:
            print('error, x should be (input_size, batch_size)')
            exit()
        elif x.ndim == 2:
            self.batch_size = x.shape[1]
        if x.shape[0] != self.input_size:
            print('error, first dim of x should be equal to DNN input_size')
            exit()
        for layer in self.layers:
            v = layer['b']


    def get_grads(self, x):
        _, loss_grads = loss



dnn = DNN()
dnn.set_input_size(2)
dnn.add_layer(4, 'tanh')
dnn.add_layer(3, 'tanh')
dnn.add_layer()
w= dnn.get_weights()

dnn.set_loss(loss_func)
dnn.set_optimizer(bfgs)
