import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.metrics import *
from src.network import Network as nn
from src.regularization import *
from src.utils.json_utils import *
from src.data_splitter import DataSplitter

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
ml_train = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TR.csv")
ml_test = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TS.csv")

# Read the data using the constructed path
x, y =  readTrainingCupData(ml_train)


internal_test_size = 0.2

data = DataSplitter(internal_test_size, random_state=None)

x, x_test, y, y_test = data.split(x, y)


n_in = np.size(x[1])
n_out = 3

init_config, train_config = load_best_model("config/ml_cup/k_fold/config_k_fold_SGD.json", model_number=1, use_train_loss=False)

network = nn(n_in, *init_config)

network.train(x, y, x_test, y_test, *train_config)

y_out = network.forward(x_test)

print(f"Loss: {MEE().compute(y_test, y_out)}")



