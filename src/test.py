import numpy as np
from metrics import *
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from utils.data_utils import *
from metrics import *
from activation_function import *



import numpy as np

script_dir = os.path.dirname(__file__)

loss = MEE()
loss_2 = MSE()

act = Act_LeakyReLU()

x = np.array([[-1, 0], [3, 2], [3, 6]])
y = np.array([[1, 2], [3, 4], [5, 6]])

# x = np.array([0, 3, 4])
# y = np.array([3, 1, 1])

print(x.ndim)

print(act.forward_fun(x))
print(act.derivative_fun(x))

# x = np.array([[2], [3], [4]])
# y = np.array([[3], [1], [1]])

# print(np.shape(x))
# print(np.shape(y))

# print(loss.compute(x, y))
# print(loss.derivative(x, y))

# differences = predictions - output
# distances = np.sqrt(np.power(differences, 2)) + epsilon

# print(x-y)

# print(np.power(x - y, 2))
# print(np.sum(np.power(x - y, 2), axis=1))
# print(np.sqrt(np.sum(np.power(x - y, 2), axis=1)))
# print(np.mean(np.sqrt(np.sum(np.power(x - y, 2), axis=1))))
# print(np.mean(np.sum(np.power(x - y, 2), axis=1)))
# print(np.sqrt(np.power(x - y, 2)))

# print(loss_2.compute(x, y))
# print(loss_2.derivative(x, y))



# Construct the relative path to the data file
monk_1_train = os.path.join(script_dir, "../data/monk+s+problems/monks-1.train")
monk_1_test = os.path.join(script_dir, "../data/monk+s+problems/monks-1.test")

# Read the data using the constructed path
x, y =  read_monk_data(monk_1_train)
x_test, y_test = read_monk_data(monk_1_test)
x = feature_one_hot_encoding(x, [3,3,2,3,4,2])
x_test = feature_one_hot_encoding(x_test, [3,3,2,3,4,2])

# print(y[0:3])
# print(np.shape(y))
