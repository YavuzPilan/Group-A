import pytest
import numpy as np

from game_utils import initialize_game_state, BOARD_SHAPE, BoardPiece, pretty_print_board, \
    clear_unwanted_characters, string_to_board, NO_PLAYER, PLAYER1, PLAYER2, connected_four


def test_initialize_game_state() -> None:
    """
    Tests if the game board is correctly initialized.

    - Ensures the board has the correct shape (BOARD_SHAPE).
    - Verifies the data type of the board elements.
    - Confirms that all positions are set to NO_PLAYER (empty).
    """
    board = initialize_game_state()
    assert board.shape == BOARD_SHAPE, "Board shape is incorrect"
    assert board.dtype == BoardPiece, "Board dtype is incorrect"
    assert np.all(board == NO_PLAYER), "Board is not initialized correctly"


def test_string_to_board() -> None:
    """
    Tests conversion from a string representation of the board to a NumPy array.

    - Compares the result of string_to_board() with the expected board state.
    - Ensures pretty_print_board() correctly restores the board string.
    """
    board_as_str = ('|==============|\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|  X       O   |\n'
                    '|==============|\n'
                    '|0 1 2 3 4 5 6 |\n')

    # Create an expected board array for verification
    expected_board = initialize_game_state()
    expected_board[0, 1] = PLAYER1  # X at column 1
    expected_board[0, 5] = PLAYER2  # O at column 5

    # Test conversion
    board = string_to_board(board_as_str)
    assert np.array_equal(board, expected_board), "Converted board does not match expected board"

    # Verify that pretty_print_board() restores the correct board format
    assert pretty_print_board(board) == board_as_str, "Pretty print does not match original board string"


def test_clear_unwanted_characters() -> None:
    """
    Tests removal of unwanted characters from the board string.

    - Ensures that only the board content remains after cleanup.
    """
    board_as_str = ('|==============|\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|  X       O   |\n'
                    '|==============|\n'
                    '|0 1 2 3 4 5 6 |\n')

    expected_result = ('       \n'
                       '       \n'
                       '       \n'
                       '       \n'
                       '       \n'
                       ' X   O \n')

    assert clear_unwanted_characters(board_as_str) == expected_result, "Character cleanup failed"


def test_connected_four() -> None:
    """
    Tests the detection of four connected pieces in different directions.

    - Checks horizontal, vertical, and diagonal connections.
    - Ensures that a non-winning state is correctly detected.
    """
    board = initialize_game_state()

    # Test horizontal connection
    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER1
    board[0, 2] = PLAYER1
    board[0, 3] = PLAYER1
    assert connected_four(board, PLAYER1) is True, "Horizontal connection failed"

    # Test vertical connection
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 0] = PLAYER1
    board[2, 0] = PLAYER1
    board[3, 0] = PLAYER1
    assert connected_four(board, PLAYER1) is True, "Vertical connection failed"

    # Test diagonal connection (top-left to bottom-right)
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 1] = PLAYER1
    board[2, 2] = PLAYER1
    board[3, 3] = PLAYER1
    assert connected_four(board, PLAYER1) is True, "Diagonal (TL-BR) connection failed"

    # Test diagonal connection (bottom-left to top-right)
    board = initialize_game_state()
    board[5, 0] = PLAYER1
    board[4, 1] = PLAYER1
    board[3, 2] = PLAYER1
    board[2, 3] = PLAYER1
    assert connected_four(board, PLAYER1) is True, "Diagonal (BL-TR) connection failed"

    # Test no connection
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER2
    board[0, 2] = PLAYER1
    board[0, 3] = PLAYER2
    assert connected_four(board, PLAYER1) is False, "False positive for non-winning state"
