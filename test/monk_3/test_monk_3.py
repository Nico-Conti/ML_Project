import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.metrics import mean_squared_error, binary_accuracy
from src.network import Network as nn
from src.utils.json_utils import *

import numpy as np

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

init_config, train_config = load_best_model("config/monk_3/config_k_fold_monk_3.json")


network = nn(n_in, *init_config)

network.train(x, y, x_test, y_true, *train_config, min_train_loss=0.057)

y_out = network.forward(x_test).flatten()

print(binary_accuracy(y_true, y_out))