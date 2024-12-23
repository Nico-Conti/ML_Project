from src.activation_function import *
import random
import itertools



def grid_search_config(param_grid, n_unit_out=1, regression=False):
    param_grid = param_grid.copy()
    n_unit_per_layer, act_per_layer  = layer_unit_act_list(param_grid, n_unit_out, regression)

    param_grid['n_unit_list'] = n_unit_per_layer
    param_grid['act_list'] = act_per_layer
    del param_grid['n_layers']
    del param_grid['a_fun']
    del param_grid['n_unit']

    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values) if len(v[-2]) == len(v[-1])]

    # print(permutations_dicts)

    return permutations_dicts

def layer_unit_act_list(param_grid, n_unit_out, regression):
    n_layers = param_grid['n_layers']

    n_unit_per_layer = []
    act_per_layer = []
    
    for n in n_layers:
    
        unit = [list(item) + [n_unit_out] for item in itertools.product(param_grid['n_unit'], repeat=n-1)]
        if regression:
            act = [list(item) + [Act_Linear()] for item in itertools.product(param_grid['a_fun'], repeat=n-1)]
        else:
            act = [list(item) + [Act_Sigmoid()] for item in itertools.product(param_grid['a_fun'], repeat=n-1)]

        n_unit_per_layer.append(unit)
        act_per_layer.append(act)
        
    n_unit_per_layer = list(itertools.chain.from_iterable(n_unit_per_layer))
    act_per_layer = list(itertools.chain.from_iterable(act_per_layer))

    # print(n_unit_per_layer)

    return n_unit_per_layer, act_per_layer


def random_serach_layer_config(param_grid, n_unit_out=1, regression=False):
    param_grid = param_grid.copy()
    n_unit_per_layer, act_per_layer  = layer_unit_act_list(param_grid, n_unit_out, regression)

    param_grid['n_unit_list'] = n_unit_per_layer
    param_grid['act_list'] = act_per_layer
    del param_grid['n_layers']
    del param_grid['a_fun']
    del param_grid['n_unit']
    del param_grid['lambd']
    del param_grid['learning_rate']
    del param_grid['momentum']
    del param_grid['patience']

    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values) if len(v[-2]) == len(v[-1])]

    return permutations_dicts

from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD

def random_search_config(param_grid, n_unit_out=1, regression=False, num_instances=1):
    layer_configs = random_serach_layer_config(param_grid, n_unit_out, regression)
    lambd_range = (0, 0.0001)
    learning_rate_range = (0.0001, 0.1)
    learning_rate_range_linear_decay_min = (0.0001, 0.001)
    learning_rate_range_linear_decay_max = (0.08, 0.4)
    linear_decay_epoch_range = (50, 150)
    momentum_range = (0.5, 0.95)

    randomized_configs = []
    
    for _ in range(num_instances):

        random_index = random.choice(range(len(layer_configs)))

        # Use the same index to select values for both
        n_unit_list = layer_configs[random_index]['n_unit_list']
        act_list = layer_configs[random_index]['act_list']

        learning_choices = [
            lr(random.uniform(*learning_rate_range)),
            lrLD(random.uniform(*learning_rate_range_linear_decay_max), random.randint(*linear_decay_epoch_range), random.uniform(*learning_rate_range_linear_decay_min))
        ]

        learning_rate = random.choice(
            learning_choices
        )

        lambd_choices = [L1(random.uniform(*lambd_range)), L2(random.uniform(*lambd_range))]

        lambd = random.choice(
            lambd_choices     
        )

        if random.random() < 0.1:
            momentum = 0
        else:
            momentum = random.uniform(*momentum_range)

        new_config = {
            'n_unit_list': n_unit_list,
            'act_list': act_list,
            'learning_rate': learning_rate,
            'lambd': lambd,
            'momentum': momentum,
            'patience': 12
        }

        randomized_configs.append(new_config)

    return randomized_configs


def parse_config(config):
    new_dict = {}
    processed_act_list = []
    
    for key, value in config.items():
        if key == 'lambd':
            class_name = value.__class__.__name__
            if hasattr(value, 'lambd'):
                new_dict[key] = f"{class_name}({value.lambd})"

        elif key == 'act_list':
            for i, act in enumerate(value):
                class_name = act.__class__.__name__
                processed_act_list.append(f"{class_name}()")
            new_dict['act_list'] = processed_act_list

        elif key == 'learning_rate':
            class_name = value.__class__.__name__
            if hasattr(value, 'tau'):
                new_dict[key] = f"{class_name}({value.learning_rate}, {value.tau}, {value.final_learning_rate})"
            else:
                new_dict[key] = f"{class_name}({value.learning_rate})"
        else:
            new_dict[key] = value
    return new_dict