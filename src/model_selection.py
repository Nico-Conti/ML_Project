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
    all_models_info = []


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
                        learning_rate=config['learning_rate'], epochs=config['epochs'],
                        lambd=config['lambd'], momentum=config['momentum'],
                        patience=config['patience'], early_stopping = True
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
                            learning_rate=config['learning_rate'], epochs=config['epochs'],
                            lambd=config['lambd'], momentum=config['momentum'],
                            patience=config['patience'], early_stopping = True
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
            
            if not np.isnan(avg_loss_val):
                best_models_info.append(current_model_info)
                current_config = current_model_info.copy()
                del current_config['k_fold_results']
                all_models_info.append(current_config)
                
            else:
                print("Nan value found in avg_val_loss, skipping this model")
            best_models_info.sort(key=lambda item: item['avg_val_loss'])
            best_models_info = best_models_info[:4]


        all_models_info.sort(key=lambda item: item['avg_val_loss'])

    top_models_for_json = []
    for model_info in best_models_info:
        json_friendly_model = {
            "config": parse_config(model_info['config']),
            "avg_val_loss": model_info['avg_val_loss'],
            "avg_train_loss": model_info['avg_train_loss'],
            "std_val_loss": model_info['std_val_loss'],  # Use .get() for optional keys
        }
        top_models_for_json.append(json_friendly_model)

    
    all_models_for_json = []
    for model_info in all_models_info:
        json_friendly_model = {
            "config": parse_config(model_info['config']),
            "avg_val_loss": model_info['avg_val_loss'],
            "avg_train_loss": model_info['avg_train_loss'],
            "std_val_loss": model_info['std_val_loss'],  # Use .get() for optional keys
        }
        all_models_for_json.append(json_friendly_model)


    return top_models_for_json, best_models_info[0], all_models_for_json