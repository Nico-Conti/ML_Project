import json
import ast
from src.activation_function import *  # Import all necessary activation functions
from src.regularization import *  # Import regularization classes
from src.learning_rate import *  # Import LearningRate class

activation_mapping = {
    'Act_ReLU()': Act_ReLU(),
    'Act_Sigmoid()': Act_Sigmoid(),
    'Act_Tanh()': Act_Tanh(),
    'Act_Linear()': Act_Linear(),
    'Act_LeakyReLU()': Act_LeakyReLU()
}

regularization_mapping = {
        'L1Regularization': L1Regularization,
        'L2Regularization': L2Regularization,
        # Add other regularization types as needed
    }

learning_rate_mapping = {
    'LearningRate': LearningRate,
    'LearningRateLinearDecay': LearningRateLinearDecay
}

def load_best_model(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    data=data[0] # Get Best Model

    model_config = data["config"]
    min_train_loss = data["avg_train_loss"]

    # Mapping for activation functions
    

    # Load activation functions
    act_list = model_config["act_list"]
    activation_functions = []
    for act in act_list:
        act_name = act.strip('"')  # Remove surrounding quotes if present
        if act_name in activation_mapping:
            activation_functions.append(activation_mapping[act_name])
        else:
            raise ValueError(f"Unknown activation function: {act_name}")

    # Load learning rate
    learning_rate_str = model_config["learning_rate"]
    # Assuming learning_rate_str is in the format 'LearningRate(0.001)'
    # Extract the value:
    learning_rate_type = learning_rate_str.split('(')[0]
    if learning_rate_type in learning_rate_mapping:
        if learning_rate_mapping[learning_rate_type] == 'LearningRateLinearDecay':
            learning_rate_max = float(learning_rate_str.split('(')[1].split(',')[0])
            learning_rate_epochs = int(learning_rate_str.split(',')[1].split(',')[0])
            learning_min = float(learning_rate_str.split(',')[1].split(')')[2])
            learning_rate = learning_rate_mapping[learning_rate_type()]
        else:
            learning_rate_value = float(learning_rate_str.split('(')[1].split(')')[0])
            learning_rate = learning_rate_mapping[learning_rate_type](learning_rate_value)
    


    

    # Load regularization
    lambd_str = model_config["lambd"]
    # Assuming lambd_str is in the format 'L1Regularization(0.01)' or 'L2Regularization(0.01)'
    lambd_type = lambd_str.split('(')[0]
    lambd_value = float(lambd_str.split('(')[1].split(')')[0])
    if lambd_type in regularization_mapping:
        lambd = regularization_mapping[lambd_type](lambd_value)
    else:
        raise ValueError(f"Unknown regularization type: {lambd_type}")

    # Load momentum
    momentum = float(model_config["momentum"])

    # Load early stopping parameters
    early_stopping = False
    patience = 0

    # Load other parameters
    batch_size = int(model_config.get("batch_size", -1))
    epochs = int(model_config.get("epochs", 500))
    min_delta = float(model_config.get("min_delta", 0.001))

    # Load n_unit_list
    n_unit_list = model_config["n_unit_list"]

    loss_function = model_config["loss_function"]

    # Create configurations
    init_config = (n_unit_list, activation_functions, loss_function)
    train_config = (batch_size, learning_rate, epochs, patience, lambd, momentum, early_stopping, min_delta, min_train_loss)

    return init_config, train_config





def prepare_json(models, k_loss_list_train):
    models_data = []

    for config in models:

        # Create a dictionary for this model
        model_dict = {
            "model": config,
        }

        models_data.append(model_dict)

    return models_data


def save_config_to_json(config, file_path):

    with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)