import numpy as np
from src.activation_function import function
from src.weight_init import init_rand_bias, init_rand_w

class LayerDense():

    def __init__(self, n_in, n_out, activation:function):
        self.weights = init_rand_w(n_in, n_out, limit=0.5, seed=2)
        self.bias = init_rand_bias(n_out, limit=0.5, seed=2)
        self.activation = activation

        self.grad_biases = np.zeros_like(self.bias)
        self.grad_weights = np.zeros_like(self.weights)


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

        if self.grad_weights.shape[1] == 1 : self.grad_weights = self.grad_weights.reshape(-1)

        self.grad_biases = np.sum(delta, axis=0) + momentum*self.grad_biases
        self.grad_weights = np.dot(self.inputs.T,delta) + momentum*self.grad_weights

        np.clip(self.grad_biases, -0.5, 0.5)
        np.clip(self.grad_weights, -0.5, 0.5)

        # print(np.shape(self.grad_weights))
        if self.weights.shape[1] == 1: self.grad_weights = self.grad_weights.reshape(self.weights.shape)
        # print(np.shape(self.grad_weights))

        if len(delta.shape) == 1: delta = delta.reshape(delta.shape[0], 1)
        new_upstream_delta = np.dot(delta, self.weights.T)

        self.bias -= self.grad_biases * learning_rate + lambd*self.bias
        self.weights -= self.grad_weights * learning_rate + lambd*self.weights

         
                

        return new_upstream_delta