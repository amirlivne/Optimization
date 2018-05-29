import numpy as np
from BFGS import BFGS


def mse_loss(y_true, y_pred, nargout=1):

    loss = (y_pred - y_true)**2
    if nargout == 2:
        grad = 2*(y_pred - y_true)
        return loss, grad
    else:
        return loss


def eval_func(x):
    return x[:, 0] * np.exp(-np.sum(x**2, axis=1))


def tanh_activation(v, nargout=1):
    val = np.tanh(v)
    if nargout == 2:
        grad = 1 - val**2
        return val, grad
    else:
        return val


def linear_activation(v, nargout=1):
    val = v
    if nargout == 2:
        grad = np.ones_like(v)
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
        self.x = None

    def set_input_size(self, size):
        self.input_size = size

    def add_layer(self, n=1, activation=linear_activation):
        if not self.layers:    # first layer
            m = self.input_size
        else:
            m = self.layers[-1]['w'].shape[1]
        w = np.random.randn(m, n) / np.sqrt(n)
        b = np.zeros((1, n))
        layer = {'w': w, 'b': b, 'activation': activation}
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.loss = loss
        self.optimizer = optimizer

    def get_weights(self):
        weights = np.empty(0)
        for layer in self.layers:
            weights = np.concatenate((weights, layer['w'].flatten(), layer['b'].flatten()))
        return weights

    def get_grads(self):
        weights = np.empty(0)
        for layer in self.layers:
            weights = np.concatenate((weights, layer['w_grads'].flatten(), layer['b_grads'].flatten()))
        return weights

    def set_weights(self, weights):
        cnt = 0
        for layer in self.layers:
            w_shape = layer['w'].shape
            w_n = np.prod(np.array(w_shape))
            layer['w'] = weights[cnt:cnt+w_n].reshape(w_shape)
            cnt += w_n

            b_shape = layer['b'].shape
            b_n = np.prod(np.array(w_shape))
            layer['b'] = weights[cnt:cnt+b_n].reshape(b_shape)
            cnt += b_n

    def train(self, x, y):
        self.x = x
        self.y = y
        weights = self.get_weights()
        weights, m = self.optimizer(self.target_func, weights)
        self.set_weights(weights)
        return m

    def target_func(self, weights):
        y_pred = self.forward(self.x)
        self.calc_grads(y_pred, self.y)
        value = np.mean(self.loss(y_pred, self.y))
        grads = self.get_grads()
        return value, grads

    def forward(self, x):  # x should be (batch_size, input_size)
        if x.ndim > 2:
            print('error, x should be (batch_size, input_size)')
            exit()
        elif x.ndim == 2:
            self.batch_size = x.shape[0]
            x = np.expand_dims(x, axis=1)
        elif x.ndim == 1:
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)

        if x.shape[2] != self.input_size:
            print('error, second dim of x should be equal to DNN input_size')
            exit()

        # self.x = x
        res = x
        for layer in self.layers:
            layer['input'] = res
            w = layer['w']
            b = layer['b']
            act = layer['activation']
            res, res_grads = act(b + np.matmul(res, w), nargout=2)
            layer['output'] = res
            layer['output_grads'] = res_grads

        return res

    def evaluate(self, x, y):
        while y.ndim < 3:
            y = np.expand_dims(y, axis=2)

        pred = self.forward(x)
        loss = np.mean(self.loss(pred, y))
        return loss

    def calc_grads(self, y_pred, y_true):
        while y_true.ndim < 3:
            y_true = np.expand_dims(y_true, axis=2)

        _, grads = self.loss(y_pred, y_true, nargout=2)
        for layer in reversed(self.layers):
            grads *= layer['output_grads']
            layer['b_grads'] = np.mean(grads, axis=0)
            layer['w_grads'] = np.mean(np.swapaxes(np.matmul(np.swapaxes(grads, 1, 2), layer['input']), 1, 2), axis=0)
            grads = np.matmul(grads, layer['w'].transpose())

