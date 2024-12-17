import os
import numpy as np

import src.activation_function as fun
from src.metrics import mean_squared_error as MSE, binary_accuracy
from src.layer import LayerDense
import src.learning_rate as lr
from src.utils.plot import provaplot, plot_learning_curve
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

    def forw_then_back(self, data_in, y_true, learning_rate, lambd, momentum):
        y_out = self.forward(data_in).flatten()
        diff = np.subtract(y_out, y_true)
        self.backward(diff, learning_rate, lambd, momentum)

    
                           
    def train(self, x_train, y_train, x_val=None, y_val=None, batch_size=-1, learning_rate=0.01, epochs=300, patience = None, lambd = 0, momentum = 0, early_stopping = True):
        if batch_size == -1:

                for epoch in range(epochs):
                    self.forw_then_back(x_train, y_train, learning_rate, lambd, momentum)
                    self.update_train_metrics(x_train, y_train)
                    self.update_val_metrics(x_val, y_val)
                    if early_stopping is True:
                        if self.wait >= patience:
                            print(f"GOOD THING THERE IS EARLY STOPPING TO SAVE THE DAY! epoch stopped at:{epoch}")
                            break

                plot_learning_curve(self.loss, self.loss_val, self.acc_val)

        else:
 
                for epoch in range(epochs):
                    for i in range(0, len(x_train), batch_size):
                        if i+batch_size < len(x_train):
                            self.forw_then_back(x_train[i:i+batch_size], y_train[i:i+batch_size], learning_rate, lambd, momentum)
                        else:
                            self.forw_then_back(x_train[i:], y_train[i:], learning_rate, lambd, momentum)
                        
                    self.update_train_metrics(x_train, y_train)
                    self.update_val_metrics(x_val, y_val)
                    if early_stopping is True:
                        if self.wait >= patience:
                            print(f"GOOD THING THERE IS EARLY STOPPING TO SAVE THE DAY! epoch stopped at:{epoch}")
                            break

                plot_learning_curve(self.loss, self.loss_val, self.acc_val)
                




    def update_train_metrics(self, data_in, y_true):
        y_out = self.forward(data_in).flatten()

        self.loss.append(MSE(y_true,y_out))
        self.acc.append(binary_accuracy(y_true,y_out))

    def update_val_metrics(self, data_in, y_true):
        y_out = self.forward(data_in).flatten()
        diff = MSE(y_true,y_out)
      
        self.loss_val.append(diff)
        self.acc_val.append(binary_accuracy(y_true,y_out))

        if diff < self.min_val:
            self.min_val = diff
            self.wait = 0
        else:
            self.wait += 1
  