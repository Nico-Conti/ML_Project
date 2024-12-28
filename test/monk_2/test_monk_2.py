import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.metrics import *
from src.network import Network as nn
from src.regularization import *
from src.utils.json_utils import *

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

init_config, train_config = load_best_model("config/monk_2/config_k_fold_monk_2.json")

network = nn(n_in, *init_config)

network.train(x, y, x_test, y_true, *train_config)

y_out = network.forward(x_test).flatten()

print(mean_euclidean_error(y_true, y_out))
print(binary_accuracy(y_true, y_out))

