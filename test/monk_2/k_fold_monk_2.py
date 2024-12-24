import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.data_splitter import *
from src.utils.plot import *
from src.network import Network as nn
from src.model_selection import *
from src.utils.hyperparameters_grid import *
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA


import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_2_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-2.train")
monk_2_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-2.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_2_train)
x_test, y_true = read_monk_data(monk_2_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_in_test = np.size(x_test[1])


val_size = 0.2
k = 5

data = k_fold(x, y, n_out, val_size, random_grid, k)


save_config_to_json(data, "config/monk_2/config_k_fold_monk_2.json")
