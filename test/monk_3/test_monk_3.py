import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.metrics import  *
from src.network import Network as nn
from src.utils.json_utils import *

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_3_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-3.train")
monk_3_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-3.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_3_train)
x_test, y_test = read_monk_data(monk_3_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

ensemble = []

for model_number in range(1, 4):
    print(f"Model number: {model_number}")

    init_config, train_config = load_best_model("config/monk_3/config_hold_monk_3.json", model_number, use_train_loss=False)

    network = nn(n_in, *init_config)

    network.train(x, y, x_test, y_test, *train_config)

    y_out = network.forward(x_test).flatten()

    ensemble.append(y_out)

    print(f"Loss: {MSE().compute(y_test, y_out)}")
    print(f"Accuracy: {binary_accuracy(y_test, y_out)}")

y_out = np.mean(ensemble, axis=0)

print(f"Loss: {MSE().compute(y_test, y_out)}")
print(f"Accuracy: {binary_accuracy(y_test, y_out)}")