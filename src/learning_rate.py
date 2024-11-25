import numpy as np

class LearningRate:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def __call__(self, epoch=0):
        return self.learning_rate

class LearningRateLinearDecay(LearningRate):
    def __init__(self, initial_learning_rate, tau, final_learning_rate):
        super().__init__(initial_learning_rate)
        self.tau = tau
        self.final_learning_rate = final_learning_rate

    def __call__(self, epoch):
        if self.tau == 0:
            return self.learning_rate
        if epoch < self.tau:
            alpha = epoch / self.tau
            return (1 - alpha) * self.learning_rate + alpha * self.final_learning_rate
        else:
            return self.final_learning_rate
        
class LearningRateExponentialDecay(LearningRate):
    def __init__(self, initial_learning_rate, decay_rate):
        super().__init__(initial_learning_rate)
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        return self.learning_rate * np.exp(-self.decay_rate * epoch)