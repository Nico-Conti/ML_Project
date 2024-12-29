from src.activation_function import  *
from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD
from src.metrics import *

grid= {
    'n_layers': [2, 3,4],
    'a_fun': [Act_ReLU(), Act_Tanh(), Act_Sigmoid(), Act_LeakyReLU()],
    'n_unit': [2, 3, 4],
    'learning_rate': [ lr(0.07), lr(0.065), lr(0.06), lr(0.058) ],
    'lambd': [L1(0.00002), L1(0.000025), L1(0.000015) ],
    'momentum': [0.95,0.92, 0.9, 0.88],
    'patience': [12]
}

random_grid_monk = {
    'n_layers': [1,1],
    'a_fun': [Act_Sigmoid(), Act_Tanh(), Act_ReLU(), Act_LeakyReLU()],
    'n_unit': [2, 6],
    'learning_rate': [0.001, 0.01],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.0000001, 0.00001],
    'momentum': [0.5, 0.95],
    'patience': [5, 10],
    'batch_size': [1, -1, 25, 50],
    'loss_function': MEE()
}


random_grid = {
    'n_layers': [1,4],
    'a_fun': [Act_Sigmoid(), Act_Tanh(), Act_ReLU(), Act_LeakyReLU()],
    'n_unit': [2, 20],
    'learning_rate': [0.001, 0.1],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.000001, 0.001],
    'momentum': [0.5, 0.95],
    'patience': [10, 20],
    'batch_size': [1, -1, 50, 100],
    'loss_function': MEE()
}

random_grid_1 = {
    'n_layers': [1,1],
    'a_fun': [ Act_ReLU()],
    'n_unit': [2, 6],
    'learning_rate': [0.001, 0.01],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.0000001, 0.00001],
    'momentum': [0.5, 0.8],
    'patience': [5, 10],
    'batch_size': [1, -1, 25, 50],
    'loss_function': MEE()
}

random_grid_2 = {
    'n_layers': [1,1],
    'a_fun': [Act_Sigmoid()],
    'n_unit': [2, 6],
    'learning_rate': [0.001, 0.01],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.0000001, 0.00001],
    'momentum': [0.5, 0.8],
    'patience': [5, 10],
    'batch_size': [1, -1, 25, 50],
    'loss_function': MEE()
}

random_grid_3 = {
    'n_layers': [1,1],
    'a_fun': [ Act_Tanh()],
    'n_unit': [2, 6],
    'learning_rate': [0.001, 0.01],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.0000001, 0.00001],
    'momentum': [0.5, 0.8],
    'patience': [5, 10],
    'batch_size': [1, -1, 25, 50],
    'loss_function': MEE()
}

random_grid_4 = {
    'n_layers': [1,1],
    'a_fun': [Act_LeakyReLU()],
    'n_unit': [2, 6],
    'learning_rate': [0.001, 0.01],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.0000001, 0.00001],
    'momentum': [0.5, 0.8],
    'patience': [5, 10],
    'batch_size': [1, -1, 25, 50],
    'loss_function': MEE()
}



fine_grid_monk_2 = {
    "n_layers": (1, 1),  # Range of number of layers
    "n_unit": (2,6,3), # (min, max, num_steps)
    "a_fun": [Act_LeakyReLU()],
    "learning_rate": (0.18, 0.32, 3),
    "learning_rate_decay_max": (0.045, 0.05, 2),
    "learning_rate_decay_epochs": (65, 75, 2),
    "learning_rate_decay_min": (0.00045, 0.0006, 3),
    "lambd": (0.000004, 0.000009, 3),
    "lambd_type": [L1],
    "momentum": (0.7, 0.9, 4),
    'patience': [10],
    'batch_size': [25, 50],
    'loss_function': [MEE()]
}

fine_grid_monk_3 = {
    "n_layers": (1, 1),  # Range of number of layers
    "n_unit": (2,5,3), # (min, max, num_steps)
    "a_fun": [Act_Tanh()],
    "learning_rate": (0.18, 0.32, 3),
    "learning_rate_decay_max": (0.045, 0.05, 3),
    "learning_rate_decay_epochs": (85, 85, 1),
    "learning_rate_decay_min": (0.0006, 0.00075, 3),
    "lambd": (0.000002, 0.000007, 3),
    "lambd_type": [L1],
    "momentum": (0.6, 0.75, 3),
    'patience': [10],
    'batch_size': [1],
    'loss_function': [MEE()]
}



