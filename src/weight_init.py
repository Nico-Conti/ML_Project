import numpy as np

# n_in -> number of neurons in input layer
# n_out ->  number of neurons in output layer


def init_forward_w(n_in, n_out, init_val=None):
    if init_val is None:
        # default 0
        init_val = 0
    return np.full((n_in, n_out), init_val)


def init_rand_w(n_in, n_out, interval=None):
    if interval is None:
        # default [-0.5, 0.5]
        limit = 0.5
        a, b = -limit, limit
    else:
        a, b = interval

    return np.random.uniform(a, b, size=(n_in, n_out))

def init_rand_bias(n_out, interval=None):
    if interval is None:
        # default [-0.5,0.5]
        limit = 0.5
        a, b = -limit, limit
    else:
        a, b = interval
    return np.random.uniform(a, b, size=n_out)


# in case of symmetric function as tanh, sigmoid ecc
def init_forw_xavier(n_in, n_out):
    w = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, w, (n_in, n_out))


def init_unif_xavier(n_in, n_out, interval=None):
    if interval is None:
        limit = np.sqrt(6 / (n_in + n_out))
        a, b = -limit, limit
    else:
        a, b = interval

    return np.random.uniform(a, b, (n_in, n_out))
