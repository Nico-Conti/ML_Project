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

    def backward(self, upstream_delta):
        # Apply the derivative of the activation function
        derivative_output = self.activation.derivative_fun(self.net)
        
        # Calculate the delta for this layer
        delta = upstream_delta * derivative_output

        self.grad_biases = delta
        self.grad_weights = np.outer(self.inputs, delta)
        
        new_upstream_delta = delta.dot(self.weights.T)
        
        return new_upstream_delta