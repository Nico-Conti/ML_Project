import os
import numpy as np

import src.act_reg_function as fun
from src.metrics import mean_squared_error as MSE, binary_accuracy
from src.layer import LayerDense
import src.learning_rate as lr
from src.utils.plot import provaplot
from src.data_splitter import DataSplitter

loss = []
acc = []

loss_val = []
acc_val = []

class Network:
    def __init__(self, n_in, nUnit_l, a_fun):
        self.layers = []
        for i in range(len(nUnit_l)):
            self.layers.append(LayerDense(n_in, nUnit_l[i], a_fun[i]))
            n_in = nUnit_l[i]

        self.min_val = float('inf')
        self.wait = 0

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
        # print(y_true-y_out)
        # print(MSE(y_true,y_out))
        self.backward(diff, learning_rate)

    def forw_val(self, data_in, y_true):
        y_out = self.forward(data_in).flatten()
        diff = np.subtract(y_out, y_true)
        diff = MSE(y_true,y_out)

        if diff < self.min_val:
            self.min_val = diff
            self.wait = 0
        else:
            self.wait += 1

        loss_val.append(MSE(y_true,y_out))
        acc_val.append(binary_accuracy(y_true,y_out))
                           
    def train(self, data_in, y_true, learning_rate, epochs=1000, batch_size=-1, patience = 20):
        if batch_size == -1:
            data = DataSplitter(val_size= 0.2)
            x_train, x_val, y_train, y_val = data.split(data_in, y_true)
            for epoch in range(epochs):
                self.forw_back(x_train, y_train, learning_rate)
                self.forw_val(x_val, y_val)

                if self.wait >= patience:
                    print(f"GOOD THING THERE IS EARLY STOPPING TO SAVE THE DAY! epoch stopped at:{epoch}")
                    break
                

            provaplot(loss, acc, epoch+1)
            provaplot(loss_val, acc_val, epoch+1)
