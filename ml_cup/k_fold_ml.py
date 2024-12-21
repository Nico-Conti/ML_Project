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

import math
import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
ml_train = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TR.csv")
ml_test = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TS.csv")

# Read the data using the constructed path
x, y =  readTrainingCupData(ml_train)


x = x[:int(len(x)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line


n_in = np.size(x[1])
n_out = 3

k_loss_list = {}

n_trials = 3
val_size = 0.2

data = DataSplitter(val_size, random_state=None)

fold_index = 1

for x_train, x_val, y_train, y_val in data.k_fold_split(x, y, k=6):
    print(len(x_train))
    print("-------------------")
    print(len(x_val))

    k_loss_list = grid_search(x_train, y_train, x_val, y_val, batch_size=-1, configs_loss=k_loss_list, n_unit_out=n_out, regression=True)
 
    fold_index += 1


for config_key, loss_list in k_loss_list.items():
    k_std = np.std(loss_list)  # Calculate the standard deviation of the loss list
    k_loss = np.mean(loss_list)  # Calculate the mean of the loss list

    k_loss_list[config_key] = [k_loss, k_std]

valid_models = {
    key: value for key, value in k_loss_list.items() if not math.isnan(value[0])
}

min_loss_config = min(valid_models, key=lambda key: k_loss_list[key][0])

min_loss = valid_models[min_loss_config][0]
std = valid_models[min_loss_config][1]

save = (f"Minimum loss: {min_loss} with std: {std} for config: {min_loss_config}")

save_config_to_json(save, "config/config_k_fold_ml.json")
