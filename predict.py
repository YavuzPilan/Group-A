import torch
import numpy as np
from projdir.model import Connect4Net


def preprocess_board(board):
    """
    Convert a (6,7) board into a (3,6,7) format for model input.
    """
    board = np.array(board).reshape(6, 7)
    board_p1 = (board == 1).astype(np.float32)  # Player 1's pieces
    board_p2 = (board == 2).astype(np.float32)  # Player 2's pieces
    board_empty = (board == 0).astype(np.float32)  # Empty spaces
    return np.stack([board_p1, board_p2, board_empty], axis=0)  # Shape: (3, 6, 7)


def load_model(model_path=r"C:\Users\pilan\Desktop\projdir\projdir\connect4_model.pth"):
    """ Load the trained model """
    model = Connect4Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model


def predict_policy_and_value(model, board):
    """
    Predict the best move (policy) and value (win probability) for a given board state.
    """
    # Preprocess board and convert to tensor
    board_tensor = torch.tensor(preprocess_board(board), dtype=torch.float32).unsqueeze(0)  # Add batch dim

    # Run model
    with torch.no_grad():
        policy_logits, value = model(board_tensor)

    # Convert policy logits to probabilities
    policy_probs = torch.softmax(policy_logits, dim=1).squeeze().numpy()

    # Get best move (highest probability)
    best_move = np.argmax(policy_probs)

    return best_move, policy_probs, value.item()


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

    print(f"Predicted Best Move: Column {best_move}")
    print(f"Move Probabilities: {policy_probs}")
    print(f"Win Probability: {value}")
