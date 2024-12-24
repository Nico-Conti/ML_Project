import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.network import Network as nn
from src.utils.plot import *
from src.utils.grid_search_configs import *
from src.data_splitter import *
from src.metrics import mean_euclidean_error as MEE
from src.metrics import binary_accuracy as BA
from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD

import numpy as np
import math
import ast

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

    print("------------------------------")
    print("Grid search In progress: ")
    print("------------------------------")

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
    return  save




def k_fold(x, y, n_out, val_size, grid, k):
    data = DataSplitter(val_size, random_state=None)

    configs = generate_random_search_configs(num_instances=10, n_unit_out=n_out, regression=False, grid = grid )

    k_loss_list_val = {}
    k_loss_list_train = {}

    fold_index = 1

    for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k=k):
        print("------------------------------")
        print("Fold In progress: ", fold_index)
        print("------------------------------")
        

        k_loss_list_val, k_loss_list_train = grid_search(
                                                x_train, y_train, x_val, y_val, batch_size=-1,
                                                configs_loss_val=k_loss_list_val,configs_loss_train=k_loss_list_train,
                                                configs=configs, fold_index=fold_index
                                                )
        
        print("------------------------------")
        print(f"Completed")
        print("------------------------------")
    
        fold_index += 1


    for config_key, loss_list in k_loss_list_val.items():
        k_std = np.std(loss_list)  # Calculate the standard deviation of the loss list
        k_loss_val = np.mean(loss_list)  # Calculate the mean of the loss list

        k_loss_list_val[config_key] = [k_loss_val, k_std]

    for config_key, loss_list in k_loss_list_train.items():
            k_loss_train = np.mean(loss_list)  # Calculate the mean of the loss list

            k_loss_list_train[config_key] = [k_loss_train]

    valid_models = {
        key: value for key, value in k_loss_list_val.items() if not math.isnan(value[0])
    }

    # min_loss_key = min(k_loss_list_val, key=lambda key: k_loss_list_val[key][0])

    sorted_models = sorted(valid_models.items(), key=lambda x: x[1][0])

    if len(sorted_models) >= 3:
        min_loss_config = sorted_models[0]  # First minimum
        second_min_loss_config = sorted_models[1]  # Second minimum
        third_min_loss_config = sorted_models[2]  # Third minimum

    # Unpack results for the first, second, and third minimums
    min_loss_val, min_std = min_loss_config[1] if min_loss_config else (None, None)
    second_min_loss_val, second_std = second_min_loss_config[1] if second_min_loss_config else (None, None)
    third_min_loss_val, third_std = third_min_loss_config[1] if third_min_loss_config else (None, None)

    # print(min_loss_config)
    min_loss_train = k_loss_list_train[min_loss_config[0]][0] if min_loss_config else None
    second_min_loss_train = k_loss_list_train[second_min_loss_config[0]][0] if second_min_loss_config else None
    third_min_loss_train = k_loss_list_train[third_min_loss_config[0]][0] if third_min_loss_config else None

    min_loss_config = ast.literal_eval(min_loss_config[0])
    act_list = []
    for act in min_loss_config['act_list']:
        act_list.append(eval(act))

    second_min_loss_config = ast.literal_eval(second_min_loss_config[0])
    act_list = []
    for act in second_min_loss_config['act_list']:
        act_list.append(eval(act))

    third_min_loss_config = ast.literal_eval(third_min_loss_config[0])
    act_list = []
    for act in third_min_loss_config['act_list']:
        act_list.append(eval(act))



    metric_1 = (f"First Eval: {min_loss_val} with std: {min_std} and train Eval: {min_loss_train}")
    metric_2 = (f"Second Eval: {second_min_loss_val} with std: {second_std} and train Eval {second_min_loss_train}")
    metric_3 = (f"Third Eval: {third_min_loss_val} with std: {third_std} and train Eval: {third_min_loss_train}")

    metric = [metric_1, metric_2, metric_3]

    data = {
        "model_1": min_loss_config,
        "model_2": second_min_loss_config,
        "model_3": third_min_loss_config,
        "metrics": metric
    }

    return data


    save_config_to_json(data, "config/config_k_fold_monk_3.json")

