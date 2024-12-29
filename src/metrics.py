import numpy as np


def accuracy(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)

def binary_accuracy(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    y_pred = np.round(y_pred).astype(int)
    
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)



class Loss():
    """ Base class for the loss functions """

    def __init__(self):
        self.name = None
    
    def __str__(self):
        return f"Loss function: {self.name}" 


class MEE(Loss):
    """ Mean Euclidean Error loss """

    def __init__(self):
        self.name = "Mean Euclidean Error"


    def compute(self, outputs, targets):  # mean euclidean error loss
        if outputs.ndim == 1:
            return np.mean(np.sqrt(np.power(targets - outputs, 2)))
        else: 
            return np.mean(np.sqrt(np.sum(np.power(targets - outputs, 2), axis=1)))
        


    def derivative(self, outputs, targets):  # derivative of MEE
        # return -2 * (targets - outputs)/(np.sqrt(np.sum(np.square(targets - outputs))))
        if outputs.ndim == 1:
            differences = targets - outputs
            return -differences / (np.sqrt(np.power(targets - outputs, 2)) + 1e-12)

        else:
            differences = targets - outputs
            norms = np.linalg.norm(differences, axis=-1, keepdims=True) + 1e-12  # Evita divisioni per zero
            return -differences / norms


class MSE(Loss):
    """ Mean Squared Error loss """

    def __init__(self):
        self.name = "Mean Squared Error"


    def compute(self, outputs, targets):  # mean squared error loss
        if outputs.ndim == 1:
            return np.mean(np.power(targets - outputs, 2))
        else:

            return np.mean(np.sum(np.power(targets - outputs, 2), axis =1))


    def derivative(self, outputs, targets):  # derivative of MSE
        return -2 * (targets - outputs)


class BinaryCrossentropy(Loss):
    """ Binary Crossentropy loss """

    def __init__(self):
        self.name = "Binary Crossentropy"

    def compute(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1 - 1e-15)  # avoids division by zero
        return np.mean(-(targets * np.log(outputs_clipped) + (1 - targets) * np.log(1 - outputs_clipped)))

    def derivative(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1 - 1e-15)  # avoids division by zero
        derivative = - (targets / outputs_clipped) + (1 - targets) / (1 - outputs_clipped)
        return np.mean(derivative, axis=0)