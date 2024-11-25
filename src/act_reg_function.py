import numpy as np


class function:
    def forward_fun(self, input_data):
        raise NotImplementedError

    def derivative_fun(self, input_data):
        raise NotImplementedError


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
        if input_data <= 0:
            return 0
        else:
            return 1


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


class Reg_Lasso(function):
    def forward_fun(self, weights, lam):
        return lam * np.sum(np.abs(weights))

    def derivative_fun(self, weights, lam):
        grad = np.zeros_like(weights)
        for i in range(len(weights)):
            if weights[i] > 0:
                grad[i] = lam
            elif weights[i] < 0:
                grad[i] = -lam
            else:
                grad[i] = 0  # could be any value in the range [-lam,+lam]

        return grad


class Reg_Ridge(function):
    def forward_fun(self, weights, lam):
        return lam * np.sum(np.square(weights))

    def derivative_fun(self, weights, lam):
        return 2 * lam * weights
