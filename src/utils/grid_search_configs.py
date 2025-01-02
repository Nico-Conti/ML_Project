from src.regularization import L2Regularization as L2
from src.regularization import L1Regularization as L1
from src.learning_rate import LearningRate as lr, LearningRateLinearDecay as lrLD
from src.activation_function import *

import random
import itertools

def generate_grid_search_configs(n_unit_out, regression, grid={}):
    param_grid = grid.copy()
    n_unit_per_layer, act_per_layer  = _generate_layer_configs(param_grid, n_unit_out, regression)

    param_grid['n_unit_list'] = n_unit_per_layer
    param_grid['act_list'] = act_per_layer
    del param_grid['n_layers']
    del param_grid['a_fun']
    del param_grid['n_unit']

    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values) if len(v[-2]) == len(v[-1])]

    return permutations_dicts

def _generate_layer_configs(param_grid, n_unit_out, regression):
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
    
    return n_unit_per_layer, act_per_layer





def generate_random_search_configs(num_instances, n_unit_out, regression, grid={}):

    randomized_configs = []

    for _ in range(num_instances):

        num_hidden_layers = random.randint(*grid["n_layers"])
        n_unit_list = [
            random.randint(*grid["n_unit"])
            for _ in range(num_hidden_layers)
        ]
        n_unit_list.append(n_unit_out)

        act_list = [
            random.choice(grid["a_fun"])
            for _ in range(num_hidden_layers)
        ]
        act_list.append(Act_Linear() if regression else Act_Sigmoid()) # Output activation


        learning_choices = [
            lr(random.uniform(*grid['learning_rate'])),
            lrLD(random.uniform(*grid['learning_rate_decay_max']), random.randint(*grid['learning_rate_decay_epochs']), random.uniform(*grid['learning_rate_decay_min']))
        ]

        learning_rate = random.choice(
            learning_choices
        )

        if random.random() < 0.15:
            lambd_choices = [L1(0), L2(0)]
        else:
            lambd_choices = [L1(random.uniform(*grid['lambd'])), L2(random.uniform(*grid['lambd']))]
        
        lambd = random.choice(
            lambd_choices     
        )

        if random.random() < 0.15:
            momentum = 0
        else:
            momentum = random.uniform(*grid['momentum'])

        patience = random.choice(grid['patience'])

        batch_size = random.choice(grid['batch_size'])

        loss_function = grid['loss_function']

        epochs = grid['epoch']

        new_config = {
            'n_unit_list': n_unit_list,
            'act_list': act_list,
            'learning_rate': learning_rate,
            'lambd': lambd,
            'momentum': momentum,
            'patience': patience,
            'batch_size': batch_size,
            'loss_function': loss_function,
            'epochs': epochs
        }

        randomized_configs.append(new_config)
        
    return randomized_configs


def generate_fine_grid_search_configs(num_instances, n_unit_out, regression, grid={}):
    """
    Generates a fine grid search over the specified hyperparameter ranges.

    Args:
        num_instances (int):  This parameter is not directly used in grid search
                             but is kept for consistency with the original function.
        n_unit_out (int): The number of output units.
        regression (bool): Whether the task is regression or classification.
        grid (dict): A dictionary specifying the ranges and number of steps for each hyperparameter.
                     Example:
                     {
                         "n_layers": (1, 3),  # Range of number of layers
                         "n_unit": (32, 128, 3), # (min, max, num_steps)
                         "a_fun": [Act_Sigmoid()],
                         "learning_rate": (0.001, 0.1, 3),
                         "learning_rate_decay_max": (0.05, 0.1, 2),
                         "learning_rate_decay_epochs": (10, 20),
                         "learning_rate_decay_min": (0.0001, 0.001, 2),
                         "lambd": (0.0001, 0.001, 2),
                         "momentum": (0.9, 0.99, 2)
                     }
    Returns:
        list: A list of dictionaries, where each dictionary represents a
              hyperparameter configuration to try.
    """
    grid_configs = []

    # Helper function to generate lists for grid search
    def generate_grid_values(key):
        if isinstance(grid[key], list):
            return grid[key]
        elif isinstance(grid[key], tuple) and len(grid[key]) >= 2:
            if len(grid[key]) == 2: # Treat as specific values if only min and max for integers
                if all(isinstance(n, int) for n in grid[key]):
                    return list(range(grid[key][0], grid[key][1] + 1))
                else:
                    return np.linspace(grid[key][0], grid[key][1], 10).tolist() # Default 10 steps for floats
            elif len(grid[key]) == 3:
                start, stop, num = grid[key]
                if isinstance(start, int) and isinstance(stop, int):
                    return np.linspace(start, stop, num, dtype=int).tolist()
                elif isinstance(start, float) or isinstance(stop, float):
                    return np.linspace(start, stop, num).tolist()
        else:
            raise ValueError(f"Invalid grid definition for '{key}'. Use a list or a tuple (min, max) or (min, max, num_steps).")

    n_layers_values = generate_grid_values("n_layers")
    n_unit_values = generate_grid_values("n_unit")
    a_fun_values = generate_grid_values("a_fun")
    lambd_type_values = generate_grid_values("lambd_type")
    lambd_values = generate_grid_values("lambd")
    patience_values = generate_grid_values("patience")
    batch_size_values = generate_grid_values("batch_size")
    loss_function_values = generate_grid_values("loss_function")

    if "learning_rate_decay_max" in grid:
        lr_decay_max_values = generate_grid_values("learning_rate_decay_max")
        lr_decay_epochs_values = generate_grid_values("learning_rate_decay_epochs")
        lr_decay_min_values = generate_grid_values("learning_rate_decay_min")

        if "momentum" in grid:
            momentum_values = generate_grid_values("momentum")
            for n_layers in n_layers_values:
                for n_unit_vals in itertools.product(n_unit_values, repeat=n_layers):
                    n_unit_list = list(n_unit_vals) + [n_unit_out]
                    for act_vals in itertools.product(a_fun_values, repeat=n_layers):
                        act_list = list(act_vals) + [Act_Linear() if regression else Act_Sigmoid()]
                        for lr_decay_max in lr_decay_max_values:
                            for lr_decay_epochs in lr_decay_epochs_values:
                                for lr_decay_min in lr_decay_min_values:
                                    learning_rate = lrLD(lr_decay_max, lr_decay_epochs, lr_decay_min)
                                    for lambd_type in lambd_type_values:
                                        for lambd_val in lambd_values:
                                            lambd = lambd_type(lambd_val)
                                            for momentum in momentum_values:
                                                for patience in patience_values:
                                                    for batch_size in batch_size_values:
                                                        for loss_function in loss_function_values:
                                                            grid_configs.append({
                                                                'n_unit_list': n_unit_list,
                                                                'act_list': act_list,
                                                                'learning_rate': learning_rate,
                                                                'lambd': lambd,
                                                                'momentum': momentum,
                                                                'patience': patience,
                                                                'batch_size': batch_size,
                                                                'loss_function': loss_function,
                                                                'epochs': grid['epoch']
                                                            })
        else:
            for n_layers in n_layers_values:
                for n_unit_vals in itertools.product(n_unit_values, repeat=n_layers):
                    n_unit_list = list(n_unit_vals) + [n_unit_out]
                    for act_vals in itertools.product(a_fun_values, repeat=n_layers):
                        act_list = list(act_vals) + [Act_Linear() if regression else Act_Sigmoid()]
                        for lr_decay_max in lr_decay_max_values:
                            for lr_decay_epochs in lr_decay_epochs_values:
                                for lr_decay_min in lr_decay_min_values:
                                    learning_rate = lrLD(lr_decay_max, lr_decay_epochs, lr_decay_min)
                                    for lambd_type in lambd_type_values:
                                        for lambd_val in lambd_values:
                                            lambd = lambd_type(lambd_val)
                                            for patience in patience_values:
                                                for batch_size in batch_size_values:
                                                    for loss_function in loss_function_values:
                                                        grid_configs.append({
                                                            'n_unit_list': n_unit_list,
                                                            'act_list': act_list,
                                                            'learning_rate': learning_rate,
                                                            'lambd': lambd,
                                                            'momentum': 0,
                                                            'patience': patience,
                                                            'batch_size': batch_size,
                                                            'loss_function': loss_function,
                                                            'epochs': grid['epoch']
                                                        })

    elif "learning_rate" in grid:
        learning_rate_values = generate_grid_values("learning_rate")
        if "momentum" in grid:
            momentum_values = generate_grid_values("momentum")
            for n_layers in n_layers_values:
                for n_unit_vals in itertools.product(n_unit_values, repeat=n_layers):
                    n_unit_list = list(n_unit_vals) + [n_unit_out]
                    for act_vals in itertools.product(a_fun_values, repeat=n_layers):
                        act_list = list(act_vals) + [Act_Linear() if regression else Act_Sigmoid()]
                        for learning_rate_val in learning_rate_values:
                            learning_rate = lr(learning_rate_val)
                            for lambd_type in lambd_type_values:
                                for lambd_val in lambd_values:
                                    lambd = lambd_type(lambd_val)
                                    for momentum in momentum_values:
                                        for patience in patience_values:
                                            for batch_size in batch_size_values:
                                                for loss_function in loss_function_values:
                                                    grid_configs.append({
                                                        'n_unit_list': n_unit_list,
                                                        'act_list': act_list,
                                                        'learning_rate': learning_rate,
                                                        'lambd': lambd,
                                                        'momentum': momentum,
                                                        'patience': patience,
                                                        'batch_size': batch_size,
                                                        'loss_function': loss_function,
                                                        'epochs': grid['epoch']
                                                    })
        else:
            for n_layers in n_layers_values:
                for n_unit_vals in itertools.product(n_unit_values, repeat=n_layers):
                    n_unit_list = list(n_unit_vals) + [n_unit_out]
                    for act_vals in itertools.product(a_fun_values, repeat=n_layers):
                        act_list = list(act_vals) + [Act_Linear() if regression else Act_Sigmoid()]
                        for learning_rate_val in learning_rate_values:
                            learning_rate = lr(learning_rate_val)
                            for lambd_type in lambd_type_values:
                                for lambd_val in lambd_values:
                                    lambd = lambd_type(lambd_val)
                                    for patience in patience_values:
                                        for batch_size in batch_size_values:
                                            for loss_function in loss_function_values:
                                                grid_configs.append({
                                                    'n_unit_list': n_unit_list,
                                                    'act_list': act_list,
                                                    'learning_rate': learning_rate,
                                                    'lambd': lambd,
                                                    'momentum': 0,
                                                    'patience': patience,
                                                    'batch_size': batch_size,
                                                    'loss_function': loss_function,
                                                    'epochs': grid['epoch']
                                                })

    return grid_configs

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

        elif key == 'loss_function':
            class_name = value.__class__.__name__
            new_dict[key] = f"{class_name}()"
        else:
            new_dict[key] = value
    return new_dict