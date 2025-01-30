import numpy as np
from src.activation_function import function
from src.weight_init import *

class LayerDense():

    def __init__(self, n_in, n_out, activation:function):
        
        # Initialize the weights and biases

        # Xavier initialization for sigmoid and tanh
        if activation.__class__.__name__ == 'Act_Sigmoid' or activation.__class__.__name__ == 'Act_Tanh' :
            self.weights = init_xavier_weights(n_in, n_out, seed=None)

        # He initialization for ReLU and LeakyReLU
        elif activation.__class__.__name__ == 'Act_LeakyReLU' or activation.__class__.__name__ == 'Act_ReLU' or activation.__class__.__name__ == 'Act_ELU':
            self.weights = init_he_weights(n_in, n_out, seed=None)
        
        # Random initialization
        else:
            self.weights = init_rand_w(n_in, n_out, limit=0.5, seed=None)
        self.bias = init_rand_bias(n_out, limit=0.5, seed=None)

        self.activation = activation

        self.grad_biases = np.zeros_like(self.bias)
        self.grad_weights = np.zeros_like(self.weights)


    def forward(self, inputs):
        self.inputs = inputs
        self.net = np.dot(inputs, self.weights) + self.bias
        self.output  = self.activation.forward_fun(self.net)

        # print(f"Inputs shape: {np.shape(self.inputs)}")

        # print(f"Weights shape: {np.shape(self.weights)}")

        # print(f"Net shape: {np.shape(self.net)}")
        # print(self.inputs)
        # print(self.weights)
        # print(self.bias)
        # print(self.net)
        # print(self.output)

        return self.output

    def backward(self, upstream_delta, learning_rate, lambd, momentum):
        
        # Apply the derivative of the activation function
        derivative_output = self.activation.derivative_fun(self.net)
        # print(derivative_output)
        if derivative_output.shape[1] == 1: derivative_output = np.reshape(derivative_output, derivative_output.shape[0])

        # Calculate the delta for this layer
        delta = upstream_delta * derivative_output

        # print(np.shape(self.grad_weights))

        if self.grad_weights.shape[1] == 1 : self.grad_weights = self.grad_weights.reshape(-1)

        # print(np.shape(self.grad_weights))

        # print(np.shape(self.inputs))
        # print(np.shape(delta))

        self.grad_biases = np.sum(delta, axis=0) + momentum*self.grad_biases
        self.grad_weights = np.dot(self.inputs.T,delta) + momentum*self.grad_weights


        np.clip(self.grad_biases, -0.5, 0.5)
        np.clip(self.grad_weights, -0.5, 0.5)

        # print(np.shape(self.grad_weights))
        if self.weights.shape[1] == 1: self.grad_weights = self.grad_weights.reshape(self.weights.shape)
        # print(np.shape(self.grad_weights))

        if len(delta.shape) == 1: delta = delta.reshape(delta.shape[0], 1)
        new_upstream_delta = np.dot(delta, self.weights.T)

        self.bias = self.bias + self.grad_biases * learning_rate - lambd(self.bias)
        self.weights = self.weights + self.grad_weights * learning_rate - lambd(self.weights) 

        return new_upstream_delta