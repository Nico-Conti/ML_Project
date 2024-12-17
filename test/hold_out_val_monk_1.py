import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.network import Network as nn
from src.utils.plot import *
from src.grid_search import grid_search
from src.data_splitter import DataSplitter
from src.metrics import mean_euclidean_error as MSE
from src.metrics import binary_accuracy as BA

import numpy as np

grid = {
    'n_layers': [2],
    'a_fun': [Act_Tanh(), Act_Sigmoid()],
    'n_unit': [2, 3, 4],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'lambd': [None],
    'momentum': [0.9, 0.7, 0.5, None],
    'patience':[9]
}

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/monk+s+problems/monks-1.train")
monk_1_test = os.path.join(script_dir, "../data/monk+s+problems/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_true = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_in_test = np.size(x_test[1])


configs = grid_search(grid)
best_loss = 1

loss_val = []
acc_val = []
 
n_trials = 4
val_size = 0.2

data = DataSplitter(val_size, random_state=None)
x_train, x_val, y_train, y_val = data.split(x, y)


for config in configs:
    
    # print(f"Trying config:{config}")
    for trial in range(n_trials):
        network = nn(n_in, config['n_unit_list'], config['act_list'])
        network.train(x_train, y_train, x_val, y_val, batch_size = 1, learning_rate=config['learning_rate'], lambd=config['lambd'], momentum=config['momentum'], patience=config['patience'], early_stopping = True)

        pred_val = network.forward(x_val).flatten()

        loss_val.append(MSE(y_val,pred_val))
        acc_val.append(BA(y_val,pred_val))

    avg_loss = sum(loss_val) / len(loss_val)
    loss_val = []
    acc_val = []

    print(avg_loss)


    if avg_loss < best_loss:
        best_loss = avg_loss
        best_config = config
        

print(f"Best config:{best_config} with val_loss={best_loss}")




# y_out = network.forward(x_test).flatten()


# print(binary_accuracy(y_true, y_out))
