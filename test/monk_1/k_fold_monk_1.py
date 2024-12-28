import sys
import os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.utils.data_utils import *
from src.utils.hyperparameters_grid import *
from src.model_selection import *
from src.utils.plot import *

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../../data/monk+s+problems/monks-1.train")
monk_1_test = os.path.join(script_dir, "../../data/monk+s+problems/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

val_size = 0.2

split_type = "stratified"
search_type = "random"   

# Define the grid
grid = random_grid_2

config, metrics = grid_search(x, y, n_in, n_out, val_size, split_type, grid, search_type, num_instances=2, regression=False, model_selection="k_fold")


for i, fold_data in enumerate(metrics['k_fold_results']):
    save_image_trials(fold_data['trial_train_losses'][0], fold_data['trial_val_losses'], fold_data['trial_val_accs'], f"config/monk_1/k_fold_monk_1_number_{i+1}.png")

save_config_to_json(config, "config/monk_1/config_k_fold_monk_1.json")


