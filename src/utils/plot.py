import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(train_losses, test_losses, accuracies):

    """
    Funzione per tracciare la learning curve basata su training loss, test loss e accuratezza.
    """
    plt.figure(figsize=(12, 6))

    if accuracies[-1] == 0:
        plt.subplot(1, 1, 1)
    else:
        plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve - Loss')
    plt.legend()

    if accuracies[-1] == 0:
        plt.show()
        return

    # Subplot per l'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve - Accuracy')
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

def plot_train_loss(train_losses):
    # import matplotlib.pyplot as plt
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="training error")
    plt.legend()
    plt.show()

def provaplot(losses, accuracies, epochs):
    # Plot loss curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accuracies, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.show()


def plot_trials(train_losses, val_losses, accuracies):
    """
    Plots the loss and accuracy curves for multiple runs, handling varying lengths.

    Parameters:
    losses (list of np.ndarray): A list where each element is an array of losses for a run.
    accuracies (list of np.ndarray): A list where each element is an array of accuracies for a run.
    """
    if len(val_losses) != len(accuracies):
        raise ValueError("The number of loss and accuracy runs must be the same.")

    num_runs = len(val_losses)

    # Set up plot
    plt.figure(figsize=(12, 5))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    for i, run_losses in enumerate(val_losses):
        plt.plot(range(1, len(run_losses) + 1), run_losses, label=f'Run {i + 1}', linestyle='--')

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    for i, run_accuracies in enumerate(accuracies):
        plt.plot(range(1, len(run_accuracies) + 1), run_accuracies, label=f'Run {i + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the plot to free memory


def save_image_trials(train_losses, val_losses, accuracies, file_path):
    """
    Saves the loss and accuracy curves for multiple runs to an image file.

    Parameters:
    losses (list of np.ndarray): A list where each element is an array of losses for a run.
    accuracies (list of np.ndarray): A list where each element is an array of accuracies for a run.
    file_path (str): The path to save the image file.
    """
    if len(val_losses) != len(accuracies):
        raise ValueError("The number of loss and accuracy runs must be the same.")

    num_runs = len(val_losses)

    # Set up plot
    plt.figure(figsize=(12, 5))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    for i, run_losses in enumerate(val_losses):
        plt.plot(range(1, len(run_losses) + 1), run_losses, label=f'Validation Loss trial {i + 1}', linestyle='--')

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    for i, run_accuracies in enumerate(accuracies):
        plt.plot(range(1, len(run_accuracies) + 1), run_accuracies, label=f'Validation Accuracy trial {i + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    # Save plot to file
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory


def save_image_folds(metrics, file_path):
    """
    Saves the loss curves for multiple folds to an image file.

    Parameters:
    losses (list of np.ndarray): A list where each element is an array of losses for a fold.
    file_path (str): The path to save the image file.
    """
    k_fold_results = metrics['k_fold_results']
    k_fold_results = k_fold_results[0]

    # Set up plot
    plt.figure(figsize=(12, 5))

    # Plot loss curve
    plt.subplot(1, 1, 1)
    for i, fold_losses in enumerate(metrics['k_fold_results']):

        plt.plot(range(1, len(fold_losses['trial_val_losses'][0]) + 1), fold_losses['trial_val_losses'][0], label=f'Fold val{i+ 1}', linestyle='--')
    plt.plot(range(1, len(k_fold_results['trial_train_losses'][0]) + 1), k_fold_results['trial_train_losses'][0], label=f'Train', linewidth=2)
        

                #  'trial_val_losses': trial_val_losses,
                # 'trial_train_losses': trial_train_losses,
                # 'trial_val_accs': trial_val_accs

    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Save plot to file
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory



def save_image_test(train_losses, test_losses, test_accuracies, file_path):
    # Set up plot
    plt.figure(figsize=(12, 5))

    if test_accuracies[-1] == 0:
        plt.subplot(1, 1, 1)
    else:
        plt.subplot(1, 2, 1)

    plt.plot(range(1, len(test_losses) + 1), test_losses, label=f'Test Loss', linestyle='--')

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

   
    num_runs = len(test_losses)

    if test_accuracies[-1] == 0:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label=f'Test accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

     # Save plot to file
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory

def save_image_val(train_losses, val_losses, val_accuracies, file_path):
    # Set up plot
    plt.figure(figsize=(12, 5))

    # Plot loss curve
    if val_accuracies:
        plt.subplot(1, 2, 1)
    else:
        plt.subplot(1, 1, 1)

    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'val Loss', linestyle='--')
    

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

   
    num_runs = len(val_losses)

    if val_accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label=f'Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

     # Save plot to file
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory
