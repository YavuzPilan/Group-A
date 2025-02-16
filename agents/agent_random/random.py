import numpy as np
from game_utils import NO_PLAYER, BoardPiece, PlayerAction, SavedState, valid_moves


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column randomly and return it as `action`
    valid_columns = valid_moves(board)

    action = np.random.choice(valid_columns)

    return action, saved_state
