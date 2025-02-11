import os
import numpy as np
import pandas as pd
import ast

# Define dataset path
from projdir.plot.plots import get_relative_path

DATASET_PATH = get_relative_path("Data", "dataset.csv")


def load_and_prepare_data(skip=1, partition_rate=0.9):
    """
    Load and preprocess the dataset efficiently, avoiding redundant operations.
    
    Args:
        skip (int): Downsample rate, e.g., `1` loads all, `2` randomly selects 50% of samples.
        partition_rate (float): Ratio of data allocated for training (rest for testing).
    
    Returns:
        tuple: Split data as NumPy arrays:
            - train_boards (array): Training set boards, shape (N, 6, 7).
            - test_boards (array): Testing set boards, shape (N', 6, 7).
            - train_policies (array): Training policy targets, shape (N, 7).
            - test_policies (array): Testing policy targets, shape (N', 7).
            - train_values (array): Training value targets, shape (N,).
            - test_values (array): Testing value targets, shape (N',).
    """
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded with {len(data)} samples.")

    # Convert Board column from string to NumPy array (Use ast.literal_eval for safety)
    print("Converting Board column to NumPy arrays...")
    boards = data['Board'].apply(ast.literal_eval).apply(np.array)
    boards = np.stack(boards.values)  # Shape: (num_samples, 6, 7)
    print(f"Boards shape: {boards.shape}")

    # Convert Policy column from string to NumPy array
    print("Converting Policy column to NumPy arrays...")
    policies = data['Policy'].apply(ast.literal_eval).apply(np.array)
    policies = np.stack(policies.values)  # Shape: (num_samples, 7)
    print(f"Policies shape: {policies.shape}")

    # Convert Value column to NumPy array
    print("Converting Value column to NumPy arrays...")
    values = data['Value'].to_numpy()  # Shape: (num_samples,)
    print(f"Values shape: {values.shape}")

    # Randomly downsample the dataset instead of fixed skipping
    if skip > 1:
        print(f"Downsampling dataset with skip rate {skip}...")
        indices = np.random.choice(len(boards), size=len(boards) // skip, replace=False)
        boards = boards[indices]
        policies = policies[indices]
        values = values[indices]
        print(f"Downsampled dataset to {len(boards)} samples.")

    # Efficient shuffling
    print("Shuffling dataset...")
    shuffle_indices = np.random.permutation(len(boards))
    boards = boards[shuffle_indices]
    policies = policies[shuffle_indices]
    values = values[shuffle_indices]
    print("Dataset shuffled.")

    # Ensure valid partition
    partition = max(1, int(len(boards) * partition_rate))
    print(f"Partitioning dataset with {partition_rate * 100}% for training...")

    # Partition dataset into training and testing sets
    train_boards, test_boards = boards[:partition], boards[partition:]
    train_policies, test_policies = policies[:partition], policies[partition:]
    train_values, test_values = values[:partition], values[partition:]

    print(f"Training set size: {len(train_boards)} samples")
    print(f"Testing set size: {len(test_boards)} samples")

    return train_boards, test_boards, train_policies, test_policies, train_values, test_values


# Example usage
if __name__ == "__main__":
    train_boards, test_boards, train_policies, test_policies, train_values, test_values = load_and_prepare_data(skip=2,
                                                                                                                partition_rate=0.8)
    print("Data preparation complete.")
