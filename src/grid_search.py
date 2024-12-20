import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.network import Network as nn
from src.utils.plot import *
from src.utils.grid_search_configs import grid_search_config
from src.data_splitter import DataSplitter
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA

import numpy as np

script_dir = os.path.dirname(__file__)

def grid_search(x_train, y_train, x_val, y_val, batch_size = -1, configs_loss={}, configs = None):
    best_loss = 1


    loss_val = []
    acc_val = []

    n_trials = 3

    n_in = np.size(x_train[1])

    for config in configs:
        for trial in range(n_trials):
            network = nn(n_in, config['n_unit_list'], config['act_list'])
            network.train(x_train, y_train, x_val, y_val, batch_size=batch_size, learning_rate=config['learning_rate'], lambd=config['lambd'], momentum=config['momentum'], patience=config['patience'], early_stopping = True)

            pred_val = network.forward(x_val).flatten()

            loss_val.append(MSE(y_val,pred_val))
            acc_val.append(BA(y_val,pred_val))

        avg_loss = sum(loss_val) / len(loss_val)
        loss_val = []
        acc_val = []

        # print(avg_loss)


        if avg_loss < best_loss:
            best_loss = avg_loss
            best_config = config

        config_key = str(config)

        if config_key not in configs_loss:
            configs_loss[config_key] = [avg_loss]
        else:
            configs_loss[config_key].append(avg_loss)

        print(configs_loss[config_key])

    print(f"Best config:{best_config} with val_loss={best_loss}")
    return configs_loss

