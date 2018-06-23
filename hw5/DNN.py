import numpy as np


def mse_loss(y_pred, y_true, nargout=1):

    loss = (y_pred - y_true)**2
    if nargout == 2:
        grad = 2*(y_pred - y_true)
        return loss, grad
    else:
        return loss


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
        self.batch_shuffle = False
        self.batch_cnt = 0

    def __get_grads__(self):
        weights = np.empty(0)
        for layer in self.layers:
            weights = np.concatenate((weights, layer['w_grads'].flatten(), layer['b_grads'].flatten()))
        return np.array(weights)

    def __calc_grads__(self, y_pred, y_true):
        while y_true.ndim < 3:
            y_true = np.expand_dims(y_true, axis=2)

        _, grads = self.loss(y_pred, y_true, nargout=2)
        for layer in reversed(self.layers):
            grads *= layer['output_grads']
            layer['b_grads'] = np.mean(grads, axis=0)
            layer['w_grads'] = np.mean(np.matmul(np.swapaxes(layer['input'], 1, 2), grads), axis=0)
            grads = np.matmul(grads, layer['w'].T)

        return self.__get_grads__()

    def __target_func__(self, weights, nargout=1):
        x, y = self.__batch_generator__()
        self.set_weights(weights)
        y_pred = self.predict(x)
        grads = self.__calc_grads__(y_pred, y)
        value = self.evaluate(x, y)
        if nargout == 1:
            return value
        return value, grads

    def __batch_generator__(self):

        k = self.batch_cnt
        b = self.batch_size
        if k == 0 and self.batch_shuffle:
            s = np.arange(self.x.shape[0])
            np.random.shuffle(s)
            self.x = self.x[s]
            self.y = self.y[s]

        if k == self.steps - 1:
            x = self.x[k*b:]
            y = self.y[k*b:]
            self.batch_cnt = 0  # avoids going too far in the data
        else:
            x = self.x[k * b:(k + 1) * b]
            y = self.y[k * b:(k + 1) * b]
            self.batch_cnt += 1  # go for the next batch

        return x, y

    def set_input_size(self, size):
        self.input_size = size

    def add_layer(self, n=1, activation=linear_activation):
        if not self.layers:    # first layer
            m = self.input_size
        else:
            m = self.layers[-1]['w'].shape[1]
        w = np.random.randn(m, n) / np.sqrt(m)
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
        return np.array(weights)

    def set_weights(self, weights):
        cnt = 0
        for layer in self.layers:
            w_shape = layer['w'].shape
            w_n = np.prod(np.array(w_shape))
            layer['w'] = np.reshape(weights[cnt:cnt+w_n], w_shape)
            cnt += w_n

            b_shape = layer['b'].shape
            b_n = np.prod(np.array(b_shape))
            layer['b'] = np.reshape(weights[cnt:cnt+b_n], b_shape)
            cnt += b_n

    def train_bfgs(self, x, y, epochs=2):
        self.x = x
        self.y = y
        self.batch_size = x.shape[0]
        self.steps = 1

        weights = self.get_weights()
        new_weights, m = self.optimizer(self.__target_func__, weights, epochs=epochs)
        self.set_weights(new_weights)
        return m

    def train(self, x, y, test=None, batch_size=None, shuffle=False, epochs=2, verbose=0):

        # init
        if test is not None:
            (x_test, y_test) = test
        self.x = x
        self.y = y
        values = []
        test_values = []
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = x.shape[0]
        self.batch_shuffle = shuffle
        self.steps = int(np.ceil(self.x.shape[0] / self.batch_size))

        # iterate epoch times
        for i in range(epochs):

            # forward and backward pass for each batch
            for j in range(self.steps):
                weights = self.get_weights()
                new_weights = self.optimizer.step(self.__target_func__, weights)
                self.set_weights(new_weights)
            values.append(self.evaluate(x, y))

            # reduce lr in the end of each epoch
            self.optimizer.epoch_end()
            if test is not None:
                test_values.append(self.evaluate(x_test, y_test))

            if verbose == 1:
                print('finished epoch ',i)
        if test is not None:
            return values, test_values
        return values

    def predict(self, x):  # x should be (batch_size, input_size)
        if x.ndim > 2:
            print('error, x should be (batch_size, input_size)')
            exit()
        elif x.ndim == 2:
            x = np.expand_dims(x, axis=1)
            # self.batch_size = x.shape[0]
        elif x.ndim == 1:
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)

        if x.shape[2] != self.input_size:
            print('error, second dim of x should be equal to DNN input_size')
            exit()

        # forward pass
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

        pred = self.predict(x)
        loss = np.mean(self.loss(pred, y))
        return loss

