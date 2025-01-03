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

data = DataSplitter(internal_test_size, random_state=42)

x, x_test, y, y_test = data.split(x, y)

k = math.ceil(1 / internal_test_size)
if k < 2:
    k = 2

test_loss = []

split_type = "split"
search_type = "random"   

# Define the grid
grid_list = [random_grid_1, random_grid_2, random_grid_3, random_grid_4]


grid = random_grid_ml

best_config, metrics, all_configs = grid_search(x, y, n_in, n_out, val_size, split_type, grid, search_type, num_instances=1000, regression=True, model_selection="k_fold")

for i, fold_data in enumerate(metrics['k_fold_results']):
    save_image_trials(fold_data['trial_train_losses'][0], fold_data['trial_val_losses'], fold_data['trial_val_accs'], f"config/ml_cup/k_fold/k_fold_ml_number_{i+1}.png")

save_config_to_json(best_config, "config/ml_cup/k_fold/config_k_fold_ml.json")
save_config_to_json(all_configs, "config/ml_cup/k_fold/configs_tried.json")


# init_config, train_config = load_best_model(f"config/ml_cup/k_fold/config_k_fold_ml.json", use_train_loss=True)

# network = nn(n_in, *init_config)
# network.train(x, y, x_test, y_test, *train_config)

# y_out = network.forward(x_test)

# train_loss_list, train_acc_list, test_loss_list, test_acc_list = network.model_metrics()


# save_image_test(train_loss_list, test_loss_list, test_acc_list, f"config/ml_cup/k_fold/hold_out_test.png")



