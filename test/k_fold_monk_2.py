import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.data_splitter import *
from src.utils.plot import *
from src.network import Network as nn
from src.grid_search import grid_search
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA


import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_2_train = os.path.join(script_dir, "../data/monk+s+problems/monks-2.train")
monk_2_test = os.path.join(script_dir, "../data/monk+s+problems/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_2_train)
x_test, y_true = read_monk_data(monk_2_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_in_test = np.size(x_test[1])

k_loss_list = {}

n_trials = 3
val_size = 0.2

data = StratifiedDataSplitter(val_size, random_state=None)

fold_index = 1

for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k=6):
    print(len(x_train))
    print("-------------------")
    print(len(x_val))

    k_loss_list = grid_search(x_train, y_train, x_val, y_val, batch_size=-1, configs_loss=k_loss_list)
 
    fold_index += 1


for config_key, loss_list in k_loss_list.items():
    k_std = np.std(loss_list)  # Calculate the standard deviation of the loss list
    k_loss = np.mean(loss_list)  # Calculate the mean of the loss list

    k_loss_list[config_key] = [k_loss, k_std]

# Find the minimum mean value in the entire k_loss dictionary
min_loss_config = min(k_loss_list, key=lambda key: k_loss_list[key][0])  # Get the key with the minimum mean loss
min_loss = k_loss_list[min_loss_config][0]
std = k_loss_list[min_loss_config][1]

save = (f"Minimum loss: {min_loss} with std: {std} for config: {min_loss_config}")

save_config_to_json(save, "config/config_k_fold_monk_2.json")
