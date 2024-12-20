from src.activation_function import *
import random
import itertools

from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1

param_grid = {
    'n_layers': [2],
    'a_fun': [Act_Tanh(), Act_Sigmoid()],
    'n_unit': [2, 3, 4],
    'learning_rate': [0.2, 0.05, 0],
    'lambd': [L2(0.01)],
    'momentum': [0.9, 0.8, 0.7, 0.6, 0.5],
    'patience':[12]
}


def grid_search_config():

    n_unit_per_layer, act_per_layer  = layer_unit_act_list(param_grid)

    param_grid['n_unit_list'] = n_unit_per_layer
    param_grid['act_list'] = act_per_layer
    del param_grid['n_layers']
    del param_grid['a_fun']
    del param_grid['n_unit']

    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values) if len(v[-2]) == len(v[-1])]

    # print(permutations_dicts)

    return permutations_dicts

def layer_unit_act_list(param_grid):
    n_layers = param_grid['n_layers']

    n_unit_per_layer = []
    act_per_layer = []
    
    for n in n_layers:
    
        unit = [list(item) + [1] for item in itertools.product(param_grid['n_unit'], repeat=n-1)]
        act = [list(item) for item in itertools.product(param_grid['a_fun'], repeat=n)]

        n_unit_per_layer.append(unit)
        act_per_layer.append(act)
        
    n_unit_per_layer = list(itertools.chain.from_iterable(n_unit_per_layer))
    act_per_layer = list(itertools.chain.from_iterable(act_per_layer))

    # print(n_unit_per_layer)

    return n_unit_per_layer, act_per_layer

def randomize_hyperp(param_grid, num_instances=3):


    randomized_configs = []
    for _ in range(num_instances):

        nUnit_l = random.choice(param_grid['nUnit_l'])

        a_fun = random.choice(param_grid['a_fun'])

        learning_rate = random.uniform(
            min(param_grid['learning_rate']), max(param_grid['learning_rate'])
        )

        lambd_values = [val for val in param_grid['lambd'] if val is not None]
        if lambd_values:
            lambd = random.uniform(min(lambd_values), max(lambd_values))
        else:
            lambd = None

        momentum_values = [val for val in param_grid['momentum'] if val is not None]
        if momentum_values:
            momentum = random.uniform(min(momentum_values), max(momentum_values))
        else:
            momentum = None

        new_config = {
            'nUnit_l': nUnit_l,
            'a_fun': a_fun,
            'learning_rate': round(learning_rate, 3),
            'lambd': None if lambd is None else round(lambd, 2),
            'momentum': None if momentum is None else round(momentum, 1)
        }
        randomized_configs.append(new_config)

    return randomized_configs