import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.data_splitter import *
from src.utils.plot import *
from src.network import Network as nn
from src.utils.grid_search_configs import *
from src.utils.hyperparameters_grid import *
from src.grid_search import grid_search
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA

import ast
import numpy as np
import math

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_3_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-3.train")
monk_3_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-3.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_3_train)
x_test, y_true = read_monk_data(monk_3_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_in_test = np.size(x_test[1])

k_loss_list_val = {}
k_loss_list_train = {}

n_trials = 3
val_size = 0.2

data = StratifiedDataSplitter(val_size, random_state=None)

configs = generate_random_search_configs(num_instances=2000, n_unit_out=1, regression=False, grid = random_grid )

fold_index = 1

for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k=5):
    print("------------------------------")
    print("Fold In progress: ", fold_index)
    print("------------------------------")
    

    k_loss_list_val, k_loss_list_train = grid_search(x_train, y_train, x_val, y_val, batch_size=-1,
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



metric_1 = (f"Minimum loss val: {min_loss_val} with std: {min_std} and min loss train: {min_loss_train}")
metric_2 = (f"Second Minimum loss val: {second_min_loss_val} with std: {second_std} and min loss train: {second_min_loss_train}")
metric_3 = (f"Third Minimum loss val: {third_min_loss_val} with std: {third_std} and min loss train: {third_min_loss_train}")

metric = [metric_1, metric_2, metric_3]

data = {
    "model_1": min_loss_config,
    "model_2": second_min_loss_config,
    "model_3": third_min_loss_config,
    "metrics": metric
}


save_config_to_json(data, "config/config_k_fold_monk_3.json")
