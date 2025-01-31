import numpy as np

# n_in -> number of neurons in input layer
# n_out ->  number of neurons in output layer


def init_forward_w(n_in, n_out, init_val=None):
    if init_val is None:
        # default 0
        init_val = 0
    return np.full((n_in, n_out), init_val)


def init_rand_w(n_in, n_out, limit=None, seed=None):
    if limit is None:
        # default [-0.5, 0.5]
        limit = 0.5
        a, b = -limit, limit
    else:
        a, b = -limit, limit

    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(a, b, size=(n_in, n_out))


def init_rand_bias(n_out, limit=None, seed=None):
    if limit is None:
        # default [-0.5,0.5]
        limit = 0.5
        a, b = -limit, limit
    else:
        a, b = -limit, limit

    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(a, b, size=n_out)


def init_xavier_weights(n_in, n_out, seed=None):
    if seed is not None:
        np.random.seed(seed)

    stddev = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0, stddev, size=(n_in, n_out))

def init_xavier_bias(n_out, seed=None):

    if seed is not None:
        np.random.seed(seed)

    return np.zeros(n_out)

def init_unif_xavier(n_in, n_out, interval=None):
    if interval is None:
        limit = np.sqrt(6 / (n_in + n_out))
        a, b = -limit, limit
    else:
        a, b = interval

    return np.random.uniform(a, b, (n_in, n_out))


def init_he_weights(n_in, n_out, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # He initialization standard deviation
    stddev = np.sqrt(2.0 / n_in)

    return np.random.normal(0, stddev, size=(n_in, n_out))


def init_he_bias(n_out, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    return np.zeros(n_out)
