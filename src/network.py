import os
import numpy as np

import src.act_reg_function as fun
from src.metrics import mean_squared_error as MSE, binary_accuracy
from src.layer import LayerDense
import src.learning_rate as lr
from src.utils.plot import provaplot

loss = []
acc = []

class Network:
    def __init__(self, n_in, nUnit_l, a_fun):
        self.layers = []
        for i in range(len(nUnit_l)):
            self.layers.append(LayerDense(n_in, nUnit_l[i], a_fun[i]))
            n_in = nUnit_l[i]

    def forward(self, data_in):
        for layer in self.layers:
            data_in = layer.forward(data_in)

        y_out = data_in
        return y_out

    def backward(self, upstream_delta, learning_rate):
        # upstream_delta is the gradient of the error of the final output
        for layer in reversed(self.layers):
            upstream_delta = layer.backward(upstream_delta, learning_rate)

    def forw_back(self, data_in, y_true, learning_rate=0.01):
        y_out = self.forward(data_in).flatten()
        diff = np.subtract(y_out, y_true)
        loss.append(MSE(y_true,y_out))
        acc.append(binary_accuracy(y_true,y_out))
        print(y_true-y_out)
        # print(MSE(y_true,y_out))
        self.backward(diff, learning_rate)

    def train(self, data_in, y_true, learning_rate, epochs=700, batch_size=-1):
        if batch_size == -1:
            for epoch in range(epochs):
                self.forw_back(data_in, y_true, learning_rate)
                

            provaplot(loss, acc, epochs)
