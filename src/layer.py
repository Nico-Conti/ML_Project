import numpy as np
from src.act_reg_function import function
from src.weight_init import init_rand_bias, init_rand_w

class LayerDense():

    def __init__(self, n_in, n_out, activation:function):
        self.weights = init_rand_w(n_in, n_out, seed=33)
        self.bias = init_rand_bias(n_out, seed=33)
        self.activation = activation

        self.grad_biases = None
        self.grad_weights = None

    def forward(self, inputs):
        self.inputs = inputs
        self.net = np.dot(inputs, self.weights) + self.bias
        self.output  = self.activation.forward_fun(self.net)

        return self.output

    def backward(self, upstream_delta, learning_rate, lambd, momentum):
        
        # Apply the derivative of the activation function
        derivative_output = self.activation.derivative_fun(self.net)
        if derivative_output.shape[1] == 1: derivative_output = np.reshape(derivative_output, derivative_output.shape[0])

        # Calculate the delta for this layer
        delta = upstream_delta * derivative_output 

        if momentum is not None and self.grad_weights is not None:
            if self.grad_weights.shape[1] == 1:
                self.grad_weights = self.grad_weights.reshape(-1)

            # print(momentum*self.grad_weights)
            # print(self.grad_weights)

            self.grad_biases = np.sum(delta, axis=0) + momentum*self.grad_biases
            self.grad_weights = np.dot(self.inputs.T,delta) + momentum*self.grad_weights
            
        else:
            self.grad_biases = np.sum(delta, axis=0)
            self.grad_weights = np.dot(self.inputs.T,delta)

        # print(np.shape(self.grad_weights))
        if self.weights.shape[1] == 1: self.grad_weights = self.grad_weights.reshape(self.weights.shape)
        # print(np.shape(self.grad_weights))

        if len(delta.shape) == 1: delta = delta.reshape(delta.shape[0], 1)
        new_upstream_delta = np.dot(delta, self.weights.T)

        if lambd is not None:
            self.bias -= self.grad_biases * learning_rate + lambd*self.bias
            self.weights -= self.grad_weights * learning_rate + lambd*self.weights
        else:
            self.bias -= self.grad_biases * learning_rate 
            self.weights -= self.grad_weights * learning_rate
            print(self.weights)

        return new_upstream_delta