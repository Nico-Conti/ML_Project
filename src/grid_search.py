import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.network import Network as nn
from src.utils.plot import *
from src.utils.grid_search_configs import *
from src.data_splitter import DataSplitter
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA
from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD

import numpy as np

script_dir = os.path.dirname(__file__)


param_grid = {
    'n_layers': [2],
    'a_fun': [Act_ReLU(), Act_Tanh(), Act_Sigmoid()],
    'n_unit': [2, 3, 4],
    'learning_rate': [ lr(0.07), lr(0.065), lr(0.06), lr(0.058) ],
    'lambd': [L1(0.00002), L1(0.000025), L1(0.000015) ],
    'momentum': [0.95,0.92, 0.9, 0.88],
    'patience': [12]
}

def grid_search(x_train, y_train, x_val, y_val, batch_size = -1, configs_loss={}, n_unit_out=1, regression=False):
    configs = grid_search_config(param_grid, n_unit_out, regression)

    # print((configs))


    loss_val = []
    acc_val = []

    n_trials = 3

    n_in = np.size(x_train[1])

    for config in configs:
        for trial in range(n_trials):
            network = nn(n_in, config['n_unit_list'], config['act_list'])
            network.train(x_train, y_train, x_val, y_val, batch_size=batch_size, learning_rate=config['learning_rate'], lambd=config['lambd'], momentum=config['momentum'], patience=config['patience'], early_stopping = True)

            pred_val = network.forward(x_val)
            if pred_val.shape[1] == 1: pred_val = np.reshape(pred_val, pred_val.shape[0])

            loss_val.append(MSE(y_val,pred_val))
            acc_val.append(BA(y_val,pred_val))

        avg_loss = sum(loss_val) / len(loss_val)
        loss_val = []
        acc_val = []

        # print(avg_loss)
        config_key = parse_config(config)
        
        config_key = str(config_key)

        if config_key not in configs_loss:
            configs_loss[config_key] = [avg_loss]
        else:
            configs_loss[config_key].append(avg_loss)

        print(configs_loss[config_key])


    return configs_loss



def random_search(x_train, y_train, x_val, y_val, batch_size = -1, configs_loss={}, randomized_configs=[]):


    loss_val = []
    acc_val = []

    n_trials = 3

    n_in = np.size(x_train[1])

    for config in randomized_configs:
        for trial in range(n_trials):
            network = nn(n_in, config['n_unit_list'], config['act_list'])
            network.train(x_train, y_train, x_val, y_val, batch_size=batch_size, learning_rate=config['learning_rate'], lambd=config['lambd'], momentum=config['momentum'], patience=config['patience'], early_stopping = True)

            pred_val = network.forward(x_val)
            if pred_val.shape[1] == 1: pred_val = np.reshape(pred_val, pred_val.shape[0])

            loss_val.append(MSE(y_val,pred_val))
            acc_val.append(BA(y_val,pred_val))

        avg_loss = sum(loss_val) / len(loss_val)
        loss_val = []
        acc_val = []

        # print(avg_loss)
        config_key = parse_config(config)
        
        config_key = str(config_key)

        if config_key not in configs_loss:
            configs_loss[config_key] = [avg_loss]
        else:
            configs_loss[config_key].append(avg_loss)

        # print(configs_loss[config_key])


    return configs_loss


