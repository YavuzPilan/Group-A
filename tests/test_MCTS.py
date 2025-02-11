import numpy as np
from game_utils import (
    NO_PLAYER, PLAYER1, PLAYER2,
)
from agents.agent_MCTS.MCTS import (
    get_valid_moves, evaluate_board, evaluate_window, MCTS, generate_move_mcts
)


def test_get_valid_moves():
    """Test that get_valid_moves correctly identifies open columns."""
    board = np.full((6, 7), NO_PLAYER)  # Empty board
    assert get_valid_moves(board) == [0, 1, 2, 3, 4, 5, 6]

    # Fill up column 3
    for row in range(6):
        board[row, 3] = PLAYER1
    assert 3 not in get_valid_moves(board), "Column 3 should not be a valid move"


def test_evaluate_board():
    """Test board evaluation heuristic."""
    board = np.full((6, 7), NO_PLAYER)
    board[0, 0] = PLAYER1  # Place a piece in the corner
    board[1, 1] = PLAYER1
    board[2, 2] = PLAYER1
    board[3, 3] = PLAYER1  # Creates a diagonal win

    assert evaluate_board(board, PLAYER1) > 0, "Evaluation should favor winning board"
    assert evaluate_board(board, PLAYER2) < 0, "Evaluation should be negative for opponent"


def test_evaluate_window():
    """Test heuristic scoring of a 4-piece window."""
    window = np.array([PLAYER1, PLAYER1, PLAYER1, NO_PLAYER])
    assert evaluate_window(window, PLAYER1) > 0, "Should favor a nearly completed sequence"
    assert evaluate_window(window, PLAYER2) < 0, "Opponent should not benefit"


def test_mcts():
    """Test that MCTS selects a valid move."""
    board = np.full((6, 7), NO_PLAYER)
    board[0, 3] = PLAYER1  # One piece in the center
    action = MCTS(board, PLAYER2, iterations=10)

    assert action in get_valid_moves(board), f"MCTS selected an invalid move: {action}"


def test_generate_move_mcts():
    """Test generate_move_mcts function."""
    board = np.full((6, 7), NO_PLAYER)
    action, _ = generate_move_mcts(board, PLAYER1, None)
    assert action in get_valid_moves(board), f"Generated move {action} is not valid"


def run_tests():
    """Run all MCTS tests."""
    test_get_valid_moves()
    test_evaluate_board()
    test_evaluate_window()
    test_mcts()
    test_generate_move_mcts()
    print("All MCTS tests passed!")


run_tests()
