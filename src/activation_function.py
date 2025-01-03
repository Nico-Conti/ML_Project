import numpy as np


class function:
    def forward_fun(self, input_data):
        raise NotImplementedError

    def derivative_fun(self, input_data):
        return None


class Act_Sigmoid(function):
    def forward_fun(self, input_data):
        return 1 / (1 + np.exp(-input_data))
 
    def derivative_fun(self, input_data):
        sigmoid = self.forward_fun(input_data)
        return sigmoid * (1 - sigmoid)


class Act_ReLU(function):
    def forward_fun(self, input_data):
        return np.maximum(input_data, 0)

    def derivative_fun(self, input_data):
        return np.where(input_data > 0, 1, 0)

class Act_LeakyReLU(function):
    def __init__(self, alpha=0.01):
        self.alpha = alpha  # Leaky slope, typically set to 0.01

    def forward_fun(self, input_data):
        # Leaky ReLU forward pass: returns alpha * input_data for negative, else input_data
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def derivative_fun(self, input_data):
        # Derivative of Leaky ReLU: alpha for negative inputs, 1 for positive inputs
        return np.where(input_data > 0, 1, self.alpha).astype(float)
    
class Act_ELU(function):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward_fun(self, input_data):
        return np.where(input_data > 0, input_data, self.alpha * (np.exp(input_data) - 1))

    def derivative_fun(self, input_data):
        elu_output = self.forward_fun(input_data)
        return np.where(input_data > 0, 1, elu_output + self.alpha)


class Act_Linear(function):
    def forward_fun(self, input_data):
        return input_data

    def derivative_fun(self, input_data):
        return np.ones(shape=np.shape(input_data))


class Act_Tanh(function):
    def forward_fun(self, input_data):
        return np.tanh(input_data)

    def derivative_fun(self, input_data):
        return 1 - np.tanh(input_data) ** 2



