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
        if np.all(input_data) <= 0:
            return np.zeros(shape=np.shape(input_data))
        else:
            return np.ones(shape=np.shape(input_data))


class Act_Linear(function):
    def forward_fun(self, input_data):
        return input_data

    def derivative_fun(self, input_data):
        return 1


class Act_Tanh(function):
    def forward_fun(self, input_data):
        return np.tanh(input_data)

    def derivative_fun(self, input_data):
        return 1 - np.tanh(input_data) ** 2


