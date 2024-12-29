import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.metrics import *
from src.network import Network as nn
from src.regularization import *
from src.utils.json_utils import *
from src.utils.plot import *

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
ml_train = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TR.csv")
ml_test = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TS.csv")

# Read the data using the constructed path
x, y =  readTrainingCupData(ml_train)
x_test =  readTestCupData(ml_test)
y_test = None

n_in = np.size(x[1])
n_out = 3


model_number = 1

ensemble = []

for outter_fold in range(1, 5):
    print(f"Model number: {model_number}")

    init_config, train_config = load_best_model(f"config/ml_cup/nested_cv/config_outter_fold_{outter_fold}.json", model_number, use_train_loss=True)

    network = nn(n_in, *init_config)

    network.train(x, y, x_test, y_test, *train_config)

    train_loss = network.loss
    # plot_train_loss(train_loss)

    y_out = network.forward(x_test)

    ensemble.append(y_out)

print(f"Individual model predictions: {ensemble}")
print("---------------------------------")
y_out = np.mean(ensemble, axis=0)
print(f"Esenmble prediction: {y_out}")