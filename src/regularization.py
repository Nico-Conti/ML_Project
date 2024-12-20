import numpy as np

class L1Regularization:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, w):
        return self.lambd * np.sign(w)
        

class L2Regularization:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, w):
        return 2 * self.lambd * w
        