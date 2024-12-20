from layer import LayerDense
from src.activation_function import Act_Sigmoid
import numpy as np

x = np.array([[1,2,3,4,5], [6,2,3,4,5]])
y = np.array([1])
n_in = 5
n_out = 1

activation = Act_Sigmoid()

# Create an instance of LayerDense
layer = LayerDense(n_in, n_out, activation)


# Perform the forward pass
output = layer.forward(x)

upstream_delta = y - output

# Now perform the backward pass
grad_inputs = layer.backward(upstream_delta)

print(grad_inputs)

