import numpy as np
from src.act_reg_function import function
from src.weight_init import init_rand_bias, init_rand_w

class LayerDense():

    def __init__(self, n_in, n_out, activation:function):
        self.weights = init_rand_w(n_in, n_out)
        self.bias = init_rand_bias(n_out)
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.net = np.dot(inputs, self.weights) + self.bias
        self.output  = self.activation.forward_fun(self.net)

        return self.output

    def backward(self, upstream_delta, learning_rate):
        
        # Apply the derivative of the activation function
        derivative_output = self.activation.derivative_fun(self.net)
        if derivative_output.shape[1] == 1: derivative_output = np.reshape(derivative_output, derivative_output.shape[0])

        # Calculate the delta for this layer
        delta = upstream_delta * derivative_output 

        self.grad_biases = np.sum(delta, axis=0)
        self.grad_weights = np.dot(self.inputs.T,delta) 
        if self.weights.shape[1] == 1: self.grad_weights = self.grad_weights.reshape(self.weights.shape)

        if len(delta.shape) == 1: delta = delta.reshape(delta.shape[0], 1)
        new_upstream_delta = np.dot(delta, self.weights.T)
        self.bias -= self.grad_biases * learning_rate
        self.weights -= self.grad_weights * learning_rate

        return new_upstream_delta