import torch
import numpy as np
from model import Connect4Net
from plot.plots import get_relative_path


def preprocess_board(board: np.ndarray) -> np.ndarray:
    """
    Convert a (6,7) Connect Four board into a (3,6,7) format for model input.

    The board is converted into three separate arrays:
    - Player 1's pieces (1's in the original board),
    - Player 2's pieces (2's in the original board),
    - Empty spaces (0's in the original board).

    The output is a stacked array of shape (3, 6, 7), suitable for the model's input.

    Args:
        board (np.ndarray): The (6,7) game board as a 2D numpy array.

    Returns:
        np.ndarray: The preprocessed board as a (3,6,7) numpy array.
    """
    board = np.array(board).reshape(6, 7)  # Ensure board is in 6x7 format
    board_p1 = (board == 1).astype(np.float32)  # Player 1's pieces
    board_p2 = (board == 2).astype(np.float32)  # Player 2's pieces
    board_empty = (board == 0).astype(np.float32)  # Empty spaces
    return np.stack([board_p1, board_p2, board_empty], axis=0)  # Shape: (3, 6, 7)


def load_model(model_path: str = get_relative_path("..","connect4_model.pth")) -> Connect4Net:
    """
    Load a trained Connect4Net model from the specified path.

    Args:
        model_path (str): Path to the trained model file. Defaults to "connect4_model.pth".

    Returns:
        Connect4Net: The loaded model, set to evaluation mode.
    """
    model = Connect4Net()  # Initialize the model
    model.load_state_dict(torch.load(model_path))  # Load model weights
    model.eval()  # Set model to evaluation mode
    return model


def predict_policy_and_value(model: Connect4Net, board: np.ndarray) -> tuple:
    """
    Predict the best move (policy) and value (win probability) for a given board state.

    The model returns:
    - The predicted policy (probabilities of the best move for each column),
    - The predicted value (win probability for the current player).

    Args:
        model (Connect4Net): The trained model to make predictions.
        board (np.ndarray): The current state of the game board as a (6,7) numpy array.

    Returns:
        tuple: A tuple containing:
            - best_move (int): The predicted best move as a column index (0 to 6).
            - policy_probs (np.ndarray): The probability distribution over all possible moves.
            - value (float): The predicted win probability for the current player.
    """
    # Preprocess board and convert to tensor
    board_tensor = torch.tensor(preprocess_board(board), dtype=torch.float32).unsqueeze(0)  # Add batch dim

    # Run model prediction without gradient computation
    with torch.no_grad():
        policy_logits, value = model(board_tensor)  # Get policy logits and value

    # Convert policy logits to probabilities
    policy_probs = torch.softmax(policy_logits, dim=1).squeeze().numpy()

    # Get best move (highest probability)
    best_move = np.argmax(policy_probs)

    return best_move, policy_probs, value.item()  # Return best move, policy probabilities, and value


if __name__ == "__main__":
    # Example board state (realistic mid-game scenario)
    example_board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0],
        [0, 0, 1, 1, 2, 0, 0],
        [0, 1, 2, 2, 1, 2, 0]
    ]

    # Load trained model
    model = load_model()

    # Predict best move and win probability
    best_move, policy_probs, value = predict_policy_and_value(model, example_board)

    # Output predictions
    print(f"Predicted Best Move: Column {best_move}")
    print(f"Move Probabilities: {policy_probs}")
    print(f"Win Probability: {value}")
