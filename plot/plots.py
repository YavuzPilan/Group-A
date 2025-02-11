import csv

import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path


def get_relative_path(*subpaths: str) -> Path:
    """
    Returns the absolute path of a file or directory relative to the script location.

    Args:
        *subpaths (str): One or more subdirectories or filenames to append to the base path.

    Returns:
        Path: The absolute path to the requested file or directory.
    """
    base_dir = Path(__file__).resolve().parent  # Get the directory of the current script
    return base_dir.joinpath(*subpaths)  # Append given subpaths


PLOTS_FOLDER = get_relative_path("All Plots")
# Ensure the directory exists
os.makedirs(PLOTS_FOLDER, exist_ok=True)


def generate_filename(plot_name):
    """
    Generates a filename with the current date and time inside the plot folder.

    Parameters:
        plot_name (str): The base name of the plot.

    Returns:
        str: A formatted filename including date and time.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(PLOTS_FOLDER, f"{plot_name}_{timestamp}.png")


# ==================== NEURAL NETWORK PLOTTING FUNCTIONS ====================

def plot_loss(epochs=None, losses=None, test_loss=None):
    """
    Plots the train and test loss curves over training epochs.

    Parameters:
        epochs (list of int, optional): Epoch numbers.
        losses (list of float, optional): Train loss values per epoch.
        test_loss (list of float, optional): Test loss values per epoch.
    """

    if epochs is None or losses is None or test_loss is None:
        # File path (update it with your actual file path)
        file_path = r"C:\Users\pilan\Desktop\projdir\projdir\Data\loss_log.csv"

        # Initialize lists
        epochs = []
        losses = []
        test_loss = []

        # Read CSV file
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                epochs.append(int(row[0]))  # Convert epoch to int
                losses.append(float(row[1]))  # Convert train loss to float
                test_loss.append(float(row[2]))  # Convert test loss to float

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linestyle='-', marker='o', color='blue', label='Train Loss')
    plt.plot(epochs, test_loss, linestyle='-', marker='s', color='red', label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train & Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(generate_filename("loss_over_epochs"))
    plt.close()


# ==================== COMPARISON PLOTS ====================


def plot_win_rate_comparison(methods, win_rates):
    """
    Compares win rates against different opponent types.

    Parameters:
        methods (list of str): Names of different opponents (e.g., "Random Bot", "MCTS", "NN").
        win_rates (list of float): Win rates in percentage.
    """

    if methods is None and win_rates is None:
        # File path (update it with your actual file path)
        file_path = get_relative_path("..", "Data", "loss_log.csv")

        # Initialize lists
        methods = []
        win_rates = []

        # Read CSV file
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                methods.append(str(row[0]))  # Convert methods to string
                win_rates.append(float(row[1]))  # Convert win_rates loss to float

    plt.figure()
    plt.bar(methods, win_rates, color=['blue', 'red', 'green'], alpha=0.7)
    plt.xlabel('Opponent Type')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate Comparison (100 Games Each)')
    plt.savefig(generate_filename("win_rate_comparison"))
    plt.close()


if __name__ == '__main__':
    plot_loss(None, None)
    # plot_win_rate_comparison(None, None)
