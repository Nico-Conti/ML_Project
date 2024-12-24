import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.data_splitter import *
from src.utils.plot import *
from src.network import Network as nn
from src.grid_search import *
from src.utils.hyperparameters_grid import *
from src.metrics import mean_euclidean_error as MEE
from src.metrics import binary_accuracy as BA
from src.learning_rate import LearningRate, LearningRateLinearDecay
from src.regularization import L2Regularization, L1Regularization

import math
import numpy as np
import ast

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
ml_train = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TR.csv")
ml_test = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TS.csv")

# Read the data using the constructed path
x, y =  readTrainingCupData(ml_train)

n_in = np.size(x[1])
n_out = 3

k_loss_list = {}
test_loss = []
min_loss_config = []

n_trials = 3
val_size = 0.2
test_size = 0.25

outter_fold = DataSplitter(test_size, random_state=None)
inner_fold = DataSplitter(val_size, random_state=None)


for x_train_out, x_test, y_train_out, y_test in outter_fold.k_fold_split(x, y, k=4):
    print("-------------------------------------------------------------------------------------")
    print("OUTTER FOLD")
    print("-------------------------------------------------------------------------------------")

    configs = generate_random_search_configs(num_instances=400, n_unit_out=3, regression=True, grid = random_grid )

    for x_train, x_val, y_train, y_val in inner_fold.k_fold_split(x_train_out, y_train_out, k=5):
        print("-------------------")
        print("INNER FOLD")
        print("-------------------")


        k_loss_list = grid_search(x_train, y_train, x_val, y_val, batch_size=-1, configs_loss=k_loss_list, configs=configs)


    for config_key, loss_list in k_loss_list.items():
        k_std = np.std(loss_list)  # Calculate the standard deviation of the loss list
        k_loss = np.mean(loss_list)  # Calculate the mean of the loss list

        k_loss_list[config_key] = [k_loss, k_std]

    valid_models = {
        key: value for key, value in k_loss_list.items() if not math.isnan(value[0])
    }

    sorted_models = sorted(valid_models.items(), key=lambda x: x[1][0])

    if len(sorted_models) >= 3:
        min_loss_config_inner = sorted_models[0]  # First minimum
        second_min_loss_config_inner = sorted_models[1]  # Second minimum
        third_min_loss_config_inner = sorted_models[2]  # Third minimum

    # Unpack results for the first, second, and third minimums
    min_loss, min_std = min_loss_config_inner[1] if min_loss_config_inner else (None, None)
    second_min_loss, second_std = second_min_loss_config_inner[1] if second_min_loss_config_inner else (None, None)
    third_min_loss, third_std = third_min_loss_config_inner[1] if third_min_loss_config_inner else (None, None)


    save = (f"Minimum loss: {min_loss} with std: {min_std} for config: {min_loss_config_inner}")

    min_loss_config_inner = ast.literal_eval(min_loss_config_inner[0])

    min_loss_config.append(save)

    act_list = []
    for act in min_loss_config_inner['act_list']:
        act_list.append(eval(act))


    network = nn(n_in, min_loss_config_inner['n_unit_list'], act_list)

    network.train(
        x, y, x_test, y_test, batch_size=-1, learning_rate=eval(min_loss_config_inner['learning_rate']),
        lambd=eval(min_loss_config_inner['lambd']), momentum = min_loss_config_inner['momentum']
    )
    
    k_loss_list = {}
    y_out = network.forward(x_test)
    if y_out.shape[1] == 1: y_out = np.reshape(y_out, y_out.shape[0])

    test_loss.append(MEE(y_test, y_out))

print(test_loss)

save_config_to_json(min_loss_config, "config/config_k_fold_ml.json")
