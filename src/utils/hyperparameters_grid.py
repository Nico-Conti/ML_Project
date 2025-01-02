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
    'loss_function': MSE()
}

random_grid_ml = {
    'n_layers': [1,1],
    'a_fun': [Act_Tanh()],
    'n_unit': [20, 30],
    'learning_rate': [0.00001, 0.01],
    'learning_rate_decay_max': [0.0001, 0.01],
    'learning_rate_decay_min': [0.00005, 0.0001],
    'learning_rate_decay_epochs': [50, 80],
    'lambd': [0.00001, 0.0001],
    'momentum': [0.5, 0.95],
    'patience': [5, 5],
    'batch_size': [-1],
    'epoch': 1000,
    'loss_function': MEE()
}

random_grid_monk_1 = {
    'n_layers': [1, 1],
    'a_fun': [Act_Tanh()],
    'n_unit': [6, 6],
    'learning_rate': [0.008, 0.03],
    'learning_rate_decay_max': [0.01, 0.03],
    'learning_rate_decay_min': [0.008, 0.008],
    'learning_rate_decay_epochs': [20, 60],
    'lambd': [0, 0],
    'momentum': [0.65, 0.75],
    'patience': [5, 10, 15],
    'batch_size': [-1],
    'loss_function': MSE()
}

random_grid_monk_3 = {
    'n_layers': [1, 2],
    'a_fun': [Act_Tanh(), Act_Sigmoid()],
    'n_unit': [2, 6],
    'learning_rate': [0.00025, 0.02],
    'learning_rate_decay_max': [0.002, 0.02],
    'learning_rate_decay_min': [0.0005, 0.0005],
    'learning_rate_decay_epochs': [20, 60],
    'lambd': [0.00001, 0.0001],
    'momentum': [0.6, 0.95],
    'patience': [5, 10, 15],
    'batch_size': [-1],
    'loss_function': MSE()
}


random_grid = {
    'n_layers': [2,4],
    'a_fun': [Act_Sigmoid(), Act_Tanh(), Act_ReLU(), Act_LeakyReLU()],
    'n_unit': [2, 20],
    'learning_rate': [0.0001, 0.1],
    'learning_rate_decay_max': [0.001, 0.05],
    'learning_rate_decay_min': [0.00001, 0.001],
    'learning_rate_decay_epochs': [50, 100],
    'lambd': [0.000001, 0.001],
    'momentum': [0.5, 0.95],
    'patience': [25, 50],
    'batch_size': [-1, 50, 100],
    'loss_function': MEE()
}

random_grid_1 = {
    'n_layers': [1,1],
    'a_fun': [ Act_ReLU()],
    'n_unit': [2, 6],
    'learning_rate': [0.001, 0.01],
    'learning_rate_decay_max': [0.01, 0.05],
    'learning_rate_decay_min': [0.000225, 0.000225],
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

fine_grid_ml= {
    "n_layers": (1, 1),  # Range of number of layers
    "n_unit": (40,40,1), # (min, max, num_steps)
    "a_fun": [Act_Sigmoid()],
    "learning_rate": (0.02, 0.025, 1),
    "learning_rate_decay_max": (0.0001, 0.0006, 5),
    "learning_rate_decay_epochs": (35, 55, 3),
    "learning_rate_decay_min": (0.00004, 0.0001, 5),
    "lambd": (0.0, 0.0, 1),
    "lambd_type": [L1],
    "momentum": (0.7, 0.75, 5),
    'patience': [10],
    'batch_size': [-1],
    'loss_function': [MEE()]
}


fine_grid_monk_1 = {
    "n_layers": (1, 1),  # Range of number of layers
    "n_unit": (6,6,1), # (min, max, num_steps)
    "a_fun": [Act_LeakyReLU()],
    "learning_rate": (0.02, 0.025, 20),
    "learning_rate_decay_max": (0.04, 0.05, 5),
    "learning_rate_decay_epochs": (40, 60, 2),
    "learning_rate_decay_min": (0.005, 0.02, 5),
    "lambd": (0, 0, 1),
    "lambd_type": [L1],
    "momentum": (0.6, 0.75, 10),
    'patience': [5],
    'batch_size': [-1],
    'loss_function': [MSE()]
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
    "n_unit": (5,5,1), # (min, max, num_steps)
    "a_fun": [Act_Tanh()],
    "learning_rate": (0.0008, 0.002, 10),
    "learning_rate_decay_max": (0.0008, 0.002, 8),
    "learning_rate_decay_epochs": (60, 60, 1),
    "learning_rate_decay_min": (0.0005, 0.00075, 5),
    "lambd": (0.00002, 0.0001, 10),
    "lambd_type": [L1],
    "momentum": (0.75, 0.75, 1),
    'patience': [10],
    'batch_size': [-1],
    'loss_function': [MSE()],
    'epoch': 300
}



