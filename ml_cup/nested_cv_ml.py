import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.data_splitter import *
from src.utils.plot import *
from src.model_selection import *
from src.utils.hyperparameters_grid import *
from src.metrics import *


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

n_trials = 3
val_size = 0.2
internal_test_size = 0.25

outter_fold = DataSplitter(internal_test_size, random_state=20)

k = math.ceil(1 / internal_test_size)
if k < 2:
    k = 2

test_loss = []

split_type = "split"
search_type = "random"   

# Define the grid
grid_list = [random_grid_1, random_grid_2, random_grid_3, random_grid_4]


outter_fold_idx = 1
for x_train, x_test, y_train, y_test in outter_fold.k_fold_split(x, y, k):
    print("------------------------------------------------------------")
    print(f"Outter fold {outter_fold_idx}")
    print("------------------------------------------------------------")
    
    # grid = grid_list[outter_fold_idx - 1]
    grid = random_grid

    config, metrics = grid_search(x_train, y_train, n_in, n_out, val_size, split_type, grid, search_type, num_instances=1000, regression=True, model_selection="k_fold")

    #Save the results of the inner k fold
    save_image_folds(metrics, f"config/ml_cup/nested_cv/outter_fold_{outter_fold_idx}.png")
    save_config_to_json(config, f"config/ml_cup/nested_cv/config_outter_fold_{outter_fold_idx}.json")

    #Training and testing model found for inner k fold on the outter fold
    init_config, train_config = load_best_model(f"config/ml_cup/nested_cv/config_outter_fold_{outter_fold_idx}.json", use_train_loss=True)

    network = nn(n_in, *init_config)
    network.train(x_train, y_train, x_test, y_test, *train_config)

    y_out = network.forward(x_test)

    train_loss_list, train_acc_list, test_loss_list, test_acc_list = network.model_metrics()

    test_loss.append(MEE().compute(y_test, y_out))

    save_image_test(train_loss_list, test_loss_list, f"config/ml_cup/nested_cv/test_outter_fold_{outter_fold_idx}.png")
    outter_fold_idx += 1

print(test_loss) 

print(f"Average loss: {np.mean(test_loss)}")


