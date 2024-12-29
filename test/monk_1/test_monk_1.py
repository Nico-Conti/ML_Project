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
monk_1_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-1.train")
monk_1_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_test = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

init_config, train_config = load_best_model("config/monk_1/config_hold_monk_1.json", model_number=1, use_train_loss=False)

print(init_config)
print(train_config)

network = nn(n_in, *init_config)

network.train(x, y, x_test, y_test, *train_config)

y_out = network.forward(x_test).flatten()

print(f"Loss: {MEE().compute(y_test, y_out)}")
print(f"Accuracy: {binary_accuracy(y_test, y_out)}")



    
