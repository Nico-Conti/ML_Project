import numpy as np

def squared_loss(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return (y_true - y_pred) ** 2

def euclidean_loss(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    return np.sqrt((y_true - y_pred) ** 2)

def mean_squared_error(y_true, y_pred):
    return np.mean(squared_loss(y_true, y_pred))

def mean_euclidean_error(y_true, y_pred):
    return np.mean(euclidean_loss(y_true, y_pred))

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

def classification_accuracy(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)
