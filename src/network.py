import os
import numpy as np

import src.activation_function as fun
from src.metrics import  binary_accuracy
from src.layer import LayerDense
from src.learning_rate import LearningRate as lr
from src.regularization import L2Regularization as L2
from src.utils.plot import provaplot, plot_learning_curve
from src.utils.data_utils import shuffle_indices
from src.data_splitter import DataSplitter
from src.metrics import MSE


class Network:
    def __init__(self, n_in, nUnit_l, a_fun, loss_function):
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

        self.loss_function = loss_function


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
        y_out = self.forward(data_in)
        if y_out.shape[1] == 1: y_out = np.reshape(y_out, y_out.shape[0])
        diff = 2 * (y_true - y_out)
        self.backward(diff, learning_rate, lambd, momentum)

    
                           
    def train(self, x_train, y_train, x_val=None, y_val=None, batch_size=-1, learning_rate=lr(0.001), epochs=300, patience = None, lambd = L2(0), momentum = 0, early_stopping = False, min_delta=0.0001, min_train_loss=0):
        if batch_size == -1:
                for epoch in range(epochs):
                    self.forw_then_back(x_train, y_train, learning_rate(epoch), lambd, momentum)
                    self.update_train_metrics(x_train, y_train)
                    if y_val is not None:
                        self.update_val_metrics(x_val, y_val)
                    if early_stopping is True:
                        self.early_stopping(min_delta)
                        if self.wait >= patience:
                            
                            # print(f"GOOD THING THERE IS EARLY STOPPING TO SAVE THE DAY! epoch stopped at:{epoch}")
                            break
                # plot_learning_curve(self.loss, self.loss_val, self.acc_val)

                    else:
                        if self.loss[-1] < min_train_loss:
                            break

                           
                if early_stopping is False:
                    plot_learning_curve(self.loss, self.loss_val, self.acc_val)

        else:
                for epoch in range(epochs):
                    x_train, y_train = shuffle_indices(x_train, y_train, random_state=None) #Random input order
                    for i in range(0, len(x_train), batch_size):
                        if i+batch_size < len(x_train):
                            self.forw_then_back(x_train[i:i+batch_size], y_train[i:i+batch_size],  learning_rate(epoch), lambd, momentum)
                        else:
                            self.forw_then_back(x_train[i:], y_train[i:],  learning_rate(epoch), lambd, momentum)
                    self.update_train_metrics(x_train, y_train)
                    if y_val is not None:
                        self.update_val_metrics(x_val, y_val)   
                    if early_stopping is True:
                        self.early_stopping(min_delta)
                        if self.wait >= patience:
                            # print(f"GOOD THING THERE IS EARLY STOPPING TO SAVE THE DAY! epoch stopped at:{epoch}")
                            break

                    else:
                        if self.loss[-1] < min_train_loss:
                            break
                
                if early_stopping is False:
                    plot_learning_curve(self.loss, self.loss_val, self.acc_val)
                




    def update_train_metrics(self, data_in, y_true):
        y_out = self.forward(data_in)
        if y_out.shape[1] == 1: y_out = np.reshape(y_out, y_out.shape[0])
        self.loss.append(self.loss_function.compute(y_true,y_out))
        self.acc.append(binary_accuracy(y_true,y_out))

    def update_val_metrics(self, data_in, y_true):
        y_out = self.forward(data_in)
        if y_out.shape[1] == 1: y_out = np.reshape(y_out, y_out.shape[0])
        self.loss_val.append(self.loss_function.compute(y_true,y_out))
        self.acc_val.append(binary_accuracy(y_true,y_out))


    def early_stopping(self, min_delta):

        if self.loss_val[-1] < self.min_val and self.min_val - self.loss_val[-1] > min_delta:
            self.min_val = self.loss_val[-1]
            self.wait = 0
        else:
            self.wait += 1

    def model_metrics(self):
        return self.loss, self.acc, self.loss_val, self.acc_val
