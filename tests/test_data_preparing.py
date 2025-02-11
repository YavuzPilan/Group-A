import numpy as np
import pandas as pd
import os
from data_preparing import load_and_prepare_data

# Define a sample dataset for testing
SAMPLE_DATA = pd.DataFrame({
    "Board": [
        "[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 0, 2, 1, 2, 0, 0]]",
        "[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 2, 1, 2, 1, 2, 1]]"
    ],
    "Policy": [
        "[0, 0, 1, 0, 0, 0, 0]",
        "[0, 1, 0, 0, 0, 0, 0]"
    ],
    "Value": [1, -1]
})

TEST_DATASET_PATH = "test_dataset.csv"


def setup_test_dataset():
    """Creates a small test dataset CSV file for testing data_preparing.py"""
    SAMPLE_DATA.to_csv(TEST_DATASET_PATH, index=False)


def test_load_and_prepare_data():
    """Test if data is correctly loaded and processed from CSV."""
    setup_test_dataset()

    # Load and preprocess the dataset
    train_boards, test_boards, train_policies, test_policies, train_values, test_values = load_and_prepare_data(skip=1,
                                                                                                                partition_rate=0.9)

    # Test if data is correctly loaded
    assert train_boards.shape[1:] == (6, 7), "Board shape should be (6, 7)"
    assert test_boards.shape[1:] == (6, 7), "Board shape should be (6, 7)"

    assert train_policies.shape[1] == 7, "Policy shape should be (N, 7)"
    assert test_policies.shape[1] == 7, "Policy shape should be (N, 7)"

    assert train_values.ndim == 1, "Values should be a 1D array"
    assert test_values.ndim == 1, "Values should be a 1D array"

    # Check if policies sum to 1 after normalization
    assert np.allclose(np.sum(train_policies, axis=1), 1), "Each policy vector should sum to 1"
