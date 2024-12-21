import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.activation_function import  *
from src.layer import LayerDense
from src.metrics import mean_squared_error, binary_accuracy
from src.network import Network as nn
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD
from src.regularization import *

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
ml_train = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TR.csv")
ml_test = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TS.csv")

# Read the data using the constructed path
x, y =  readTrainingCupData(ml_train)


x = x[:int(len(x)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line

x_test = x[int(len(x)*0.75):]
y_test = y[int(len(y)*0.75):] 


n_in = np.size(x[1])
n_out = 3

n_unit_per_layer = [4, 1]
act_per_layer = [Act_LeakyReLU(), Act_ReLU()]

network = nn(n_in, n_unit_per_layer, act_per_layer)

network.train(x, y, x_test, y_test, batch_size=-1, patience = 12, learning_rate=lrLD(0.01, 50, 0.001), lambd=L1Regularization(0), momentum = 0,  early_stopping=False)

y_out = network.forward(x_test).flatten()

print(mean_squared_error(y_test, y_out))
print(binary_accuracy(y_test, y_out))

