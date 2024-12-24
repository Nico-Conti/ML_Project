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


import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-1.train")
monk_1_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
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

fold_index = 1

configs = configs = generate_random_search_configs(num_instances=300, n_unit_out=1, regression=False, grid = random_grid )

for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k=5):
    print("------------------------------")
    print("Fold In progress: ", fold_index)
    print("------------------------------")
    

    k_loss_list_val, k_loss_list_train = grid_search(x_train, y_train, x_val, y_val, batch_size=-1,
                configs_loss_val=k_loss_list_val,configs_loss_train=k_loss_list_train, configs=configs
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

# Find the minimum mean value in the entire k_loss dictionary
min_loss_config = min(k_loss_list_val, key=lambda key: k_loss_list_val[key][0])  # Get the key with the minimum mean loss
min_loss_val = k_loss_list_val[min_loss_config][0]
min_loss_train = k_loss_list_train[min_loss_config][0]
std = k_loss_list_val[min_loss_config][1]



save = (f"Minimum loss val: {min_loss_val} with std: {std} and min loss train: {min_loss_train} for config: {min_loss_config}")

save_config_to_json(save, "config/config_k_fold_monk_1.json")
