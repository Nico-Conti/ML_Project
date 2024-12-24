from src.activation_function import  *
from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD

grid= {
    'n_layers': [2, 3,4],
    'a_fun': [Act_ReLU(), Act_Tanh(), Act_Sigmoid(), Act_LeakyReLU()],
    'n_unit': [2, 3, 4],
    'learning_rate': [ lr(0.07), lr(0.065), lr(0.06), lr(0.058) ],
    'lambd': [L1(0.00002), L1(0.000025), L1(0.000015) ],
    'momentum': [0.95,0.92, 0.9, 0.88],
    'patience': [12]
}


random_grid = {
    'n_layers': [1,2],
    'a_fun': [Act_Sigmoid(), Act_Tanh(), Act_LeakyReLU(), Act_ReLU()],
    'n_unit': [2, 6],
    'learning_rate': [0.01, 0.05],
    'learning_rate_decay_max': [0.01, 0.2],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.0000001, 0.00001],
    'momentum': [0.5, 0.8],
    'patience': [5]
}

fine_grid = {
    "n_layers": (1, 2),  # Range of number of layers
    "n_unit": (2,5,3), # (min, max, num_steps)
    "a_fun": [Act_LeakyReLU(), Act_Sigmoid(), Act_ReLU()],
    "learning_rate": (0.18, 0.32, 3),
    "learning_rate_decay_max": (0.0058, 0.06, 1),
    "learning_rate_decay_epochs": (80, 90, 2),
    "learning_rate_decay_min": (0.0009, 0.001, 1),
    "lambd": (0.0001, 0.001, 3),
    "lambd_type": [L1],
    "momentum": (0.91, 0.925, 3)
}

random_grid_2 = {
    'n_layers': [1,2],
    'a_fun': [Act_Sigmoid(), Act_ReLU()],
    'n_unit': [3, 5],
    'learning_rate': [0.05, 0.25],
    'learning_rate_decay_max': [0.01, 0.2],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.001, 0.01],
    'momentum': [0.85, 0.95],
    'patience': [10, 20]
}
