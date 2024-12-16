import os
import numpy as np

import src.activation_function as fun
from src.metrics import mean_squared_error as MSE, binary_accuracy
from src.layer import LayerDense
import src.learning_rate as lr
from src.utils.plot import provaplot
from src.data_splitter import DataSplitter


class Network:
    def __init__(self, n_in, nUnit_l, a_fun):
        self.layers = []
        for i in range(len(nUnit_l)):
            self.layers.append(LayerDense(n_in, nUnit_l[i], a_fun[i]))
            n_in = nUnit_l[i]

        self.min_val = float('inf')
        self.wait = 0

        self.loss = []
        self.acc = []
        self.loss_val = []
        self.acc_val = []

    def forward(self, data_in):
        for layer in self.layers:
            data_in = layer.forward(data_in)

        y_out = data_in
        return y_out

    def backward(self, upstream_delta, learning_rate, lambd, momentum):
        # upstream_delta is the gradient of the error of the final output
        for layer in reversed(self.layers):
            upstream_delta = layer.backward(upstream_delta, learning_rate, lambd, momentum)

    def forw_back(self, data_in, y_true, learning_rate, lambd, momentum):
        y_out = self.forward(data_in).flatten()
        diff = np.subtract(y_out, y_true)
        self.loss.append(MSE(y_true,y_out))
        self.acc.append(binary_accuracy(y_true,y_out))
        # print(y_true-y_out)
        # print(MSE(y_true,y_out))
        self.backward(diff, learning_rate, lambd, momentum)

    def forw_val(self, data_in, y_true):
        y_out = self.forward(data_in).flatten()
        diff = np.subtract(y_out, y_true)
        diff = MSE(y_true,y_out)

        if diff < self.min_val:
            self.min_val = diff
            self.wait = 0
        else:
            self.wait += 1

        self.loss_val.append(MSE(y_true,y_out))
        self.acc_val.append(binary_accuracy(y_true,y_out))
                           
    def train(self, x_train, y_train, x_val=None, y_val=None, learning_rate=0.01, epochs=500, batch_size=-1, patience = None, lambd = None, momentum = None, early_stopping = True):
        if batch_size == -1:
            
            if early_stopping is True:

                for epoch in range(epochs):
                    self.forw_back(x_train, y_train, learning_rate, lambd, momentum)
                    self.forw_val(x_val, y_val)
                    if self.wait >= patience:
                        # print(f"GOOD THING THERE IS EARLY STOPPING TO SAVE THE DAY! epoch stopped at:{epoch}")
                        break

                # provaplot(self.loss_val, self.acc_val, epoch+1)
                # provaplot(self.loss, self.acc, epoch+1)

            else:
                for epoch in range(epochs):
                    self.forw_back(x_train, y_train, learning_rate, lambd, momentum)
     
                
                provaplot(self.loss, self.acc, epoch+1)
 

