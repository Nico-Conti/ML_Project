import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  Act_Sigmoid, Act_Tanh, Act_LeakyReLU
from src.layer import LayerDense
from src.metrics import mean_squared_error, binary_accuracy
from src.network import Network as nn
from src.utils.plot import *
from src.regularization import L1Regularization as L1

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_3_train = os.path.join(script_dir, "../data/monk+s+problems/monks-3.train")
monk_3_test = os.path.join(script_dir, "../data/monk+s+problems/monks-3.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_3_train)
x_test, y_true = read_monk_data(monk_3_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_in_test = np.size(x_test[1])

n_unit_per_layer = [3,1]
act_per_layer = [Act_LeakyReLU(), Act_Sigmoid()]

network = nn(n_in, n_unit_per_layer, act_per_layer)

network.train(x, y, x_test, y_true, learning_rate=0.1, batch_size=-1, lambd=L1(0.00001), momentum=0.95, early_stopping=False)

y_out = network.forward(x_test).flatten()


print(binary_accuracy(y_true, y_out))