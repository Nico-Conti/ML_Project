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