import numpy as np
from game_utils import NO_PLAYER, BoardPiece, PlayerAction, SavedState

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column randomly and return it as `action`
    valid_columns = [col for col in range(board.shape[1]) if board[0, col] == NO_PLAYER]
    
    action = np.random.choice(valid_columns)
    
    return action, saved_state
    