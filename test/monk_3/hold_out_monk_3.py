import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.utils.hyperparameters_grid import *
from src.model_selection import *
from src.utils.plot import *


import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_3_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-3.train")
monk_3_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-3.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_3_train)
x_test, y_true = read_monk_data(monk_3_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1


val_size = 0.2

split_type = "stratified"
search_type = "random"   

# Define the grid
grid = random_grid_monk_3



best_config, metrics, all_configs = grid_search(x, y, n_in, n_out, val_size, split_type, grid, search_type, num_instances=4000, regression=False, model_selection="hold_out", seed_split=42)

save_image_val(metrics['trial_train_losses'][0], metrics['trial_val_losses'][0], metrics['trial_val_accs'][0], "config/monk_3/hold_out_monk_3.png")

save_config_to_json(best_config, "config/monk_3/config_hold_monk_3.json")
save_config_to_json(all_configs, "config/monk_3/all_config_hold_monk_3.json")

