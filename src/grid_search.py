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
from src.metrics import mean_euclidean_error as MEE
from src.metrics import binary_accuracy as BA
from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD

import numpy as np

script_dir = os.path.dirname(__file__)


def grid_search(x_train, y_train, x_val, y_val, batch_size = -1, configs_loss_val={}, configs_loss_train={},  configs=[], fold_index = 0):
    # configs = grid_search_config(n_unit_out, regression)

    total_configs = len(configs)

    loss_train = []
    loss_val = []
    acc_val = []

    n_trials = 3

    n_in = np.size(x_train[1])

    for i, config in enumerate(configs):
        print(f"\nProcessing configuration {i+1}/{total_configs} ---------- Fold {fold_index} ----------")
        for trial in range(n_trials):
            network = nn(n_in, config['n_unit_list'], config['act_list'])
            network.train(x_train, y_train, x_val, y_val, batch_size=batch_size, learning_rate=config['learning_rate'], lambd=config['lambd'], momentum=config['momentum'], patience=config['patience'], early_stopping = True)

            pred_val = network.forward(x_val)
            if pred_val.shape[1] == 1: pred_val = np.reshape(pred_val, pred_val.shape[0])
            pred_train = network.forward(x_train)
            if pred_train.shape[1] == 1: pred_train = np.reshape(pred_train, pred_train.shape[0])

            loss_val.append(MEE(y_val,pred_val))
            acc_val.append(BA(y_val,pred_val))
            loss_train.append(MEE(y_train,pred_train))

        avg_loss_val = sum(loss_val) / len(loss_val)
        avg_loss_train = sum(loss_train) / len(loss_train)
        loss_val = []
        loss_train = []
        acc_val = []

        # print(avg_loss)
        config_key = parse_config(config)
        
        config_key = str(config_key)

        if config_key not in configs_loss_val:
            configs_loss_val[config_key] = [avg_loss_val]
        else:
            configs_loss_val[config_key].append(avg_loss_val)

        if config_key not in configs_loss_train:
            configs_loss_train[config_key] = [avg_loss_train]
        else:
            configs_loss_train[config_key].append(avg_loss_train)

        # print(configs_loss[config_key])


    return configs_loss_val, configs_loss_train


def hold_out_validation(x, y, n_out, val_size, grid):

    data = DataSplitter(val_size, random_state=None)

    x_train, x_val, y_train, y_val = data.split(x, y)

    configs_loss_val = {}
    configs_loss_train = {}

    configs = generate_random_search_configs(num_instances=1, n_unit_out=n_out, regression=False, grid=grid)

    config_loss_val, config_loss_train = grid_search(
                                    x_train, y_train, x_val, y_val, batch_size=-1,
                                    configs_loss_val=configs_loss_val,
                                    configs_loss_train = configs_loss_train,
                                    configs=configs
                                )

    min_loss_config = min(config_loss_val, key=config_loss_val.get)

    eval = (f"Eval: {config_loss_val[min_loss_config]}  for config: {min_loss_config}")
    train_loss = (f"Eval train: {config_loss_train[min_loss_config]}  for config: {min_loss_config}")

    save = [eval , train_loss]

    # print(min_loss_config)
    save_config_to_json(save, "config/config_hold_monk_1.json")

