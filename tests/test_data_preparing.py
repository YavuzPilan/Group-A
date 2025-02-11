import numpy as np
import pandas as pd
from data_preparing import load_and_prepare_data

# Define a sample dataset for testing
SAMPLE_DATA: pd.DataFrame = pd.DataFrame({
    "Board": [
        # Each board state is represented as a stringified list of lists (6x7 grid)
        "[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, "
        "0, 0], [1, 0, 2, 1, 2, 0, 0]]",
        "[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, "
        "0, 0], [1, 2, 1, 2, 1, 2, 1]] "
    ],
    "Policy": [
        # Policy vector representing the probability distribution for each possible move
        "[0, 0, 1, 0, 0, 0, 0]",
        "[0, 1, 0, 0, 0, 0, 0]"
    ],
    "Value": [1, -1]  # Game outcome: 1 for win, -1 for loss
})

# Path to store the temporary test dataset CSV file
TEST_DATASET_PATH: str = "test_dataset.csv"


def setup_test_dataset() -> None:
    """
    Creates a small test dataset CSV file for testing data_preparing.py.

    This function saves the SAMPLE_DATA DataFrame to a CSV file
    to be used in testing the data loading and processing functionality.
    """
    SAMPLE_DATA.to_csv(TEST_DATASET_PATH, index=False)


def test_load_and_prepare_data() -> None:
    """
    Test if data is correctly loaded and processed from CSV.

    The function first creates a test dataset using setup_test_dataset(),
    then loads and prepares the data using load_and_prepare_data().

    Assertions check:
    - The board dimensions match the expected shape (6x7).
    - Policy vectors have the correct shape (N, 7).
    - Value arrays are one-dimensional.
    - The sum of each policy vector is approximately 1 after normalization.
    """
    setup_test_dataset()  # Generate test dataset

    # Load and preprocess the dataset
    train_boards, test_boards, train_policies, test_policies, train_values, test_values = load_and_prepare_data(
        skip=1, partition_rate=0.9
    )

    # Verify that board dimensions are correct (6 rows, 7 columns)
    assert train_boards.shape[1:] == (6, 7), "Board shape should be (6, 7)"
    assert test_boards.shape[1:] == (6, 7), "Board shape should be (6, 7)"

    # Verify that policy vectors have the correct shape (each row has 7 values)
    assert train_policies.shape[1] == 7, "Policy shape should be (N, 7)"
    assert test_policies.shape[1] == 7, "Policy shape should be (N, 7)"

    # Ensure that value arrays are one-dimensional
    assert train_values.ndim == 1, "Values should be a 1D array"
    assert test_values.ndim == 1, "Values should be a 1D array"

    # Verify that each policy vector sums to approximately 1 (after normalization)
    assert np.allclose(np.sum(train_policies, axis=1), 1), "Each policy vector should sum to 1"
