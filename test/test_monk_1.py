import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.act_reg_function import  Act_Sigmoid, Act_Tanh
from src.layer import LayerDense
from src.metrics import mean_squared_error
from src.network import Network as nn

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
data_path = os.path.join(script_dir, "../data/monk+s+problems/monks-1.train")


# Read the data using the constructed path
x, y =  read_monk_data(data_path)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])

n_in = np.size(x[1])
n_out = 1

n_unit_per_layer = [4,6,1]
act_per_layer = [Act_Tanh(), Act_Sigmoid(), Act_Sigmoid()]

network = nn(n_in, n_unit_per_layer, act_per_layer)

network.train(x,y,0.01)

# Create an instance of LayerDense
# layer = LayerDense(n_in, n_out, activation)

# Perform training
# for i in range(3000):
#     output = layer.forward(x).flatten() 
#     diff = np.subtract(output, y)
#     layer.backward(diff)

#     print(mean_squared_error(y, output))
    
    
