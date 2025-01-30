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

init_config, train_config = load_best_model("config/monk_3/config_hold_monk_3.json", model_number=1)

avg_test_loss = []
avg_test_accuracy = []
avg_train_accuracy = []
avg_train_loss = []


#Up range if you want to have a average score of model
for i in range(0, 1):
    network = nn(n_in, *init_config)

    network.train(x, y, x_test, y_test, *train_config, plot=True)

    y_out = network.forward(x_test).flatten()
    y_pred = network.forward(x).flatten()

    avg_test_loss.append(MSE().compute(y_test, y_out))
    avg_test_accuracy.append(binary_accuracy(y_test, y_out))
    avg_train_loss.append(MSE().compute(y, y_pred))
    avg_train_accuracy.append(binary_accuracy(y, y_pred))

    
print(f"Average test loss: {np.mean(avg_test_loss)}")
print(f"Average test accuracy: {np.mean(avg_test_accuracy)}")
print(f"Average train loss: {np.mean(avg_train_loss)}")
print(f"Average train accuracy: {np.mean(avg_train_accuracy)}")