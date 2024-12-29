import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.network import Network as nn
from src.network import *
from src.utils.grid_search_configs import *
from src.data_splitter import *
from src.utils.json_utils import *

import numpy as np
import math


script_dir = os.path.dirname(__file__)


# def grid_search(x_train, y_train, x_val, y_val, batch_size = -1, configs_loss_val={}, configs_loss_train={},  configs=[], fold_index = 1):

#     total_configs = len(configs)

#     loss_train = []
#     loss_val = []
#     acc_val = []

#     n_trials = 3

#     n_in = np.size(x_train[1])

#     for i, config in enumerate(configs):

        
#         print(f"\nProcessing configuration {i+1}/{total_configs} ---------- Fold {fold_index} ----------")
#         for trial in range(n_trials):
#             network = nn(n_in, config['n_unit_list'], config['act_list'])
#             network.train(x_train, y_train, x_val, y_val, batch_size=batch_size, learning_rate=config['learning_rate'], lambd=config['lambd'], momentum=config['momentum'], patience=config['patience'], early_stopping = True)

#             pred_val = network.forward(x_val)
#             if pred_val.shape[1] == 1: pred_val = np.reshape(pred_val, pred_val.shape[0])
#             pred_train = network.forward(x_train)
#             if pred_train.shape[1] == 1: pred_train = np.reshape(pred_train, pred_train.shape[0])

#             loss_val.append(MEE(y_val,pred_val))
#             acc_val.append(BA(y_val,pred_val))
#             loss_train.append(MEE(y_train,pred_train))


#         avg_loss_val = sum(loss_val) / len(loss_val)
#         avg_loss_train = sum(loss_train) / len(loss_train)
#         loss_val = []
#         loss_train = []
#         acc_val = []

#         # print(avg_loss)
#         config_key = parse_config(config)

#         config_key = str(config_key)

#         if config_key not in configs_loss_val:
#             configs_loss_val[config_key] = [avg_loss_val]
#         else:
#             configs_loss_val[config_key].append(avg_loss_val)

#         if config_key not in configs_loss_train:
#             configs_loss_train[config_key] = [avg_loss_train]
#         else:
#             configs_loss_train[config_key].append(avg_loss_train)

#         # print(configs_loss[config_key])


#     return configs_loss_val, configs_loss_train


def hold_out_validation(x, y, n_out, val_size, split_type, grid, search_type, num_instances=10, regression=False):

    if split_type == "stratified":
        data = StratifiedDataSplitter(val_size, random_state=None)
    else:
        data = DataSplitter(val_size, random_state=None)

    if search_type == "random":
        configs = generate_random_search_configs(num_instances, n_out, regression, grid)

    elif search_type == "fine":
        configs = generate_fine_grid_search_configs(num_instances, n_out, regression, grid)

    else:
        print("Search type not mentioned performing random search")
        configs = generate_random_search_configs(num_instances, n_out, regression, grid)
        return

    x_train, x_val, y_train, y_val = data.split(x, y)

    configs_loss_val = {}
    configs_loss_train = {}

    print("------------------------------")
    print("Grid search In progress: ")
    print("------------------------------")

    config_loss_val, config_loss_train = grid_search(
                                    x_train, y_train, x_val, y_val, batch_size=-1,
                                    configs_loss_val=configs_loss_val,
                                    configs_loss_train = configs_loss_train,
                                    configs=configs
                                )
    
    print("------------------------------")
    print(f"Completed")
    print("------------------------------")

    # Find the mean and standard deviation of the loss values for each configuration
    for config_key, loss_list in config_loss_val.items():
        std = np.std(loss_list)  # Calculate the standard deviation of the loss list
        loss_val = np.mean(loss_list)  # Calculate the mean of the loss list

        config_loss_val[config_key] = [loss_val, std]

    for config_key, loss_list in config_loss_train.items():
            loss_train = np.mean(loss_list)  # Calculate the mean of the loss list

            config_loss_train[config_key] = [loss_train]

    valid_models = {
        key: value for key, value in config_loss_val.items() if not math.isnan(value[0])
    }


    sorted_models = sorted(valid_models.items(), key=lambda x: x[1][0])

    split = [x_train, y_train, x_val, y_val]

    return prepare_json(sorted_models, config_loss_train), split



def k_fold(x, y, n_out, val_size, split_type, grid, search_type, num_instances=10, regression=False):

    if split_type == "stratified":
        data = StratifiedDataSplitter(val_size, random_state=None)
    else:
        data = DataSplitter(val_size, random_state=None)

    if search_type == "random":
        configs = generate_random_search_configs(num_instances, n_out, regression, grid)

    elif search_type == "fine":
        configs = generate_fine_grid_search_configs(num_instances, n_out, regression, grid)

    else:
        print("Search type not mentioned performing random search")
        configs = generate_random_search_configs(num_instances, n_out, regression, grid)

    k_loss_list_val = {}
    k_loss_list_train = {}

    fold_index = 1

    k = math.ceil(1 / val_size)
    if k < 2:
        k = 2

    for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k):
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

    sorted_models = sorted(valid_models.items(), key=lambda x: x[1][0])

    return prepare_json(sorted_models, k_loss_list_train)


def grid_search(x, y, n_in, n_out, val_size, split_type, grid, search_type, num_instances=10, regression=False, model_selection = "hold_out"):

    n_trials = 3

    if split_type == "stratified":
        data = StratifiedDataSplitter(val_size, random_state=None)
    else:
        data = DataSplitter(val_size, random_state=None)

    if search_type == "random":
        configs = generate_random_search_configs(num_instances, n_out, regression, grid)

    elif search_type == "fine":
        configs = generate_fine_grid_search_configs(num_instances, n_out, regression, grid)

    else:
        print("Search type not mentioned performing random search")
        configs = generate_random_search_configs(num_instances, n_out, regression, grid)

    total_configs = len(configs)

    best_loss_val = float('inf')
    best_models_info = []


    for i, config in enumerate(configs):
        print("------------------------------")
        print(f"\nProcessing configuration {i+1}/{total_configs}")
        # print("------------------------------")

        
        if model_selection == "hold_out":
            trial_val_losses = []
            trial_train_losses = []
            trial_val_accs = []
            
            x_train, x_val, y_train, y_val = data.split(x, y)
            
            for trial in range(n_trials):
                network = nn(n_in, config['n_unit_list'], config['act_list'], config['loss_function'])

                network.train(
                        x_train, y_train, x_val, y_val, batch_size=config['batch_size'],
                        learning_rate=config['learning_rate'], lambd=config['lambd'],
                        momentum=config['momentum'], patience=config['patience'], early_stopping = True
                    )
                
                train_loss, train_acc, val_loss, val_acc = network.model_metrics()
                trial_val_losses.append(val_loss)
                trial_train_losses.append(train_loss)
                trial_val_accs.append(val_acc)

            avg_loss_val = np.mean([tl[-1] for tl in trial_val_losses])
            avg_loss_train = np.mean([tl[-1] for tl in trial_train_losses])
            std_loss_val = np.std([tl[-1] for tl in trial_val_losses])

            current_model_info = {
                'config': config,
                'avg_val_loss': avg_loss_val,
                'avg_train_loss': avg_loss_train,
                'std_val_loss': std_loss_val,
                'trial_val_losses': trial_val_losses,
                'trial_train_losses': trial_train_losses,
                'trial_val_accs': trial_val_accs
            }

            best_models_info.append(current_model_info)
            best_models_info.sort(key=lambda item: item['avg_val_loss'])

            best_models_info = best_models_info[:3]
            

        elif model_selection == "k_fold":
            k = math.ceil(1 / val_size)
            if k < 2:
                k = 2

            fold_index = 1

            k_fold_results = []
            for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k):
                # print("------------------------------")
                # print("Fold In progress: ", fold_index)
                # print("------------------------------")

                trial_val_loss = []
                trial_train_loss = []
                trial_val_acc = []


                for trial in range(n_trials):
                    network = nn(n_in, config['n_unit_list'], config['act_list'], config['loss_function'])

                    network.train(
                            x_train, y_train, x_val, y_val, batch_size=config['batch_size'],
                            learning_rate=config['learning_rate'], lambd=config['lambd'],
                            momentum=config['momentum'], patience=config['patience'], early_stopping = True
                    )

                    train_loss, train_acc, val_loss, val_acc = network.model_metrics()


                    trial_val_loss.append(val_loss)
                    trial_train_loss.append(train_loss)
                    trial_val_acc.append(val_acc)
                
                k_fold_results.append({
                    'trial_val_losses': trial_val_loss,
                    'trial_train_losses': trial_train_loss,
                    'trial_val_accs': trial_val_acc
                })

                fold_index += 1
                
            avg_loss_val = np.mean([np.mean([tl[-1] for tl in fold['trial_val_losses']]) for fold in k_fold_results])
            avg_loss_train = np.mean([np.mean([tl[-1] for tl in fold['trial_train_losses']]) for fold in k_fold_results])
            std_loss_val = np.std([np.mean([tl[-1] for tl in fold['trial_val_losses']]) for fold in k_fold_results])

            current_model_info = {
                'config': config,
                'avg_val_loss': avg_loss_val,
                'avg_train_loss': avg_loss_train,
                'std_val_loss': std_loss_val,
                'k_fold_results': k_fold_results
            }
            
            best_models_info.append(current_model_info)
            best_models_info.sort(key=lambda item: item['avg_val_loss'])
            best_models_info = best_models_info[:3]

    top_models_for_json = []
    for model_info in best_models_info:
        json_friendly_model = {
            "config": parse_config(model_info['config']),
            "avg_val_loss": model_info['avg_val_loss'],
            "avg_train_loss": model_info['avg_train_loss'],
            "std_val_loss": model_info['std_val_loss'],  # Use .get() for optional keys
        }
        top_models_for_json.append(json_friendly_model)


    return top_models_for_json, best_models_info[0]