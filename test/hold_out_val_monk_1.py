import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.network import Network as nn
from src.utils.plot import *

from src.data_splitter import DataSplitter
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA
from src.grid_search import grid_search
from src.utils.grid_search_configs import grid_search_config

import numpy as np


script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/monk+s+problems/monks-1.train")
monk_1_test = os.path.join(script_dir, "../data/monk+s+problems/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_in_test = np.size(x_test[1])

n_trials = 1
val_size = 0.2

data = DataSplitter(val_size, random_state=None)

x_train, x_val, y_train, y_val = data.split(x, y)

configs = grid_search_config()
grid_search(x_train, y_train, x_val, y_val, batch_size=-1, configs=configs)


