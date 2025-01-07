import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from src.utils.data_utils import *
from src.metrics import *
from src.network import Network as nn
from src.regularization import *
from src.utils.json_utils import *
from src.utils.plot import *

import numpy as np

script_dir = os.path.dirname(__file__)

# Construct the relative path to the data file
ml_train = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TR.csv")
ml_test = os.path.join(script_dir, "../data/ml_cup/ML-CUP24-TS.csv")

# Read the data using the constructed path
x, y =  readTrainingCupData(ml_train)
x_test =  readTestCupData(ml_test)
y_test = None

n_in = np.size(x[1])
n_out = 3


model_number = 1

ensemble = []

for outter_fold in range(1, 5):
    print(f"Model : {outter_fold}")

    init_config, train_config = load_best_model(f"config/ml_cup/nested_cv/fine_grid_search/config_outter_fold_{outter_fold}.json", model_number, use_train_loss=True)

    network = nn(n_in, *init_config)

    network.train(x, y, x_test, y_test, *train_config)

    train_loss = network.loss
    # plot_train_loss(train_loss)

    y_out = network.forward(x_test)

    ensemble.append(y_out)

# print(f"Individual model predictions: {ensemble}")
# print("---------------------------------")
y_out = np.mean(ensemble, axis=0)
# print(f"Esenmble prediction: {y_out}")

def save_predictions(y_pred, folder = "."):
    # Assicurati che y_pred abbia 3 colonne
    if y_pred.shape[1] != 3:
        raise ValueError(f"Le predizioni devono avere esattamente 3 colonne (out_x, out_y, out_z). Trovato: {y_pred.shape[1]}")

    # Prepara gli ID e combina i dati
    ids = np.arange(1, y_pred.shape[0] + 1).reshape(-1, 1)  # ID da 1 a n interi
    results = np.hstack((ids, y_pred))

    # Prepara l'header personalizzato
    header = [
        "# Conti Nico, Federico Bonaccorsi",
        "# Praise the ELU",
        "# ML-CUP24 v1",
        "# 22 Jan 2024"
    ]

    # Nome del file di output
    output_file = f"{folder}/Praise_the_ELU_ML-CUP24-TS.csv"

    # Scrive il file CSV con l'header personalizzato
    with open(output_file, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, results, fmt="%d,%f,%f,%f", delimiter=",", newline="\n")

    return os.path.abspath(output_file)



save_predictions(y_out)