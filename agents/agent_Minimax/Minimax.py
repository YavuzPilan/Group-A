import numpy as np

from game_utils import check_end_state, GameState, NO_PLAYER, PLAYER1, PLAYER2, apply_player_action, BoardPiece, \
    PlayerAction, SavedState

MAX_DEPTH = 1000
SCORE4 = 500
SCORE3 = 100
SCORE2 = 10


def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple[
    float, PlayerAction]:
    """
    Perform the minimax algorithm with alpha-beta pruning, prioritizing winning moves.

    Parameters:
    - board (np.ndarray): Current game board state.
    - depth (int): Current depth in the game tree.
    - alpha (float): Alpha value for pruning.
    - beta (float): Beta value for pruning.
    - maximizing_player (bool): True if the current move is for the maximizing player, else False.

    Returns:
    - tuple[float, PlayerAction]: The best score and column for the move.
    """
    valid_moves = [col for col in range(board.shape[1]) if board[-1, col] == NO_PLAYER]
    is_terminal = check_end_state(board, PLAYER1) != GameState.STILL_PLAYING or \
                  check_end_state(board, PLAYER2) != GameState.STILL_PLAYING or \
                  len(valid_moves) == 0

    if depth == 0 or is_terminal:
        if check_end_state(board, PLAYER1) == GameState.IS_WIN:
            return (float('inf'), None) if maximizing_player else (float('-inf'), None)
        elif check_end_state(board, PLAYER2) == GameState.IS_WIN:
            return (float('-inf'), None) if maximizing_player else (float('inf'), None)
        elif check_end_state(board, PLAYER1) == GameState.IS_DRAW:
            return (0, None)
        else:
            return (evaluate_board(board, PLAYER1 if maximizing_player else PLAYER2), None)

    player = PLAYER1 if maximizing_player else PLAYER2
    for col in valid_moves:
        temp_board = board.copy()
        apply_player_action(temp_board, col, player)
        if check_end_state(temp_board, player) == GameState.IS_WIN:
            return (float('inf') if maximizing_player else float('-inf'), col)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = np.random.choice(valid_moves)
        for col in valid_moves:
            temp_board = board.copy()
            apply_player_action(temp_board, col, PLAYER1)
            eval, _ = minimax(temp_board, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = col
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = np.random.choice(valid_moves)
        for col in valid_moves:
            temp_board = board.copy()
            apply_player_action(temp_board, col, PLAYER2)
            eval, _ = minimax(temp_board, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = col
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


def evaluate_board(board: np.ndarray, player: BoardPiece) -> float:
    """
    Heuristic evaluation function for the board to assess advantage.

    Parameters:
    - board (np.ndarray): The board to evaluate.
    - player (BoardPiece): The player to evaluate advantage for.

    Returns:
    - float: A score representing the board's favorability for `player`.
    """
    opponent = PLAYER1 if player == PLAYER2 else PLAYER2
    score = 0

    score += count_windows(board, player, 4) * SCORE4
    score += count_windows(board, player, 3) * SCORE3
    score += count_windows(board, player, 2) * SCORE2

    score -= count_windows(board, opponent, 4) * SCORE4
    score -= count_windows(board, opponent, 3) * SCORE3
    score -= count_windows(board, opponent, 2) * SCORE2

    return score


def count_windows(board: np.ndarray, player: BoardPiece, num_pieces: int) -> int:
    """
    Count the number of windows (rows, columns, diagonals) that contain exactly
    `num_pieces` of `player` pieces and empty spaces elsewhere.

    Parameters:
    - board (np.ndarray): The board to check.
    - player (BoardPiece): The player piece to check for.
    - num_pieces (int): Number of `player` pieces in the window.

    Returns:
    - int: The count of such windows.
    """
    count = 0
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            # Horizontal
            if col <= board.shape[1] - 4:
                window = board[row, col:col + 4]
                if np.count_nonzero(window == player) == num_pieces and np.count_nonzero(
                        window == NO_PLAYER) == 4 - num_pieces:
                    count += 1
            # Vertical
            if row <= board.shape[0] - 4:
                window = board[row:row + 4, col]
                if np.count_nonzero(window == player) == num_pieces and np.count_nonzero(
                        window == NO_PLAYER) == 4 - num_pieces:
                    count += 1
            # Positive Diagonal
            if row <= board.shape[0] - 4 and col <= board.shape[1] - 4:
                window = [board[row + i, col + i] for i in range(4)]
                if np.count_nonzero(window == player) == num_pieces and np.count_nonzero(
                        window == NO_PLAYER) == 4 - num_pieces:
                    count += 1
            # Negative Diagonal
            if row >= 3 and col <= board.shape[1] - 4:
                window = [board[row - i, col + i] for i in range(4)]
                if np.count_nonzero(window == player) == num_pieces and np.count_nonzero(
                        window == NO_PLAYER) == 4 - num_pieces:
                    count += 1
    return count


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Generate a move using the minimax algorithm with alpha-beta pruning.

    Parameters:
    - board (np.ndarray): Current game board state.
    - player (BoardPiece): The player making the move (PLAYER1 or PLAYER2).
    - saved_state (SavedState | None): Optional saved state.

    Returns:
    - tuple[PlayerAction, SavedState | None]: The chosen action and updated saved state.
    """
    maximizing_player = player == PLAYER1
    _, action = minimax(board, MAX_DEPTH, float('-inf'), float('inf'), maximizing_player)
    return action, saved_state
