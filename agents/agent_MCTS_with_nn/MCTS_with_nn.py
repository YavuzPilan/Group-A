import copy
import math
from game_utils import valid_moves, apply_player_action, check_end_state, GameState, PLAYER1, PLAYER2
from predict import predict_policy_and_value, load_model  # Import neural network functions


class MCTSNode_with_nn:
    def __init__(self, board, parent=None, move=None, player=None, prior=1.0):
        """
        Initializes the MCTS node with neural network guidance.
        """
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = move
        self.player = player
        self.untried_moves = valid_moves(board)
        self.prior = prior  # Neural network move probability
        self.value = 0  # Neural network value estimate

    def uct_value_with_nn(self, exploration_param=1.4):
        """
        Computes UCT value incorporating prior probabilities.
        """
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_param * self.prior * math.sqrt(
            math.log(self.parent.visits) / self.visits)

    def best_child_with_nn(self, exploration_param=1.4):
        """
        Selects the best child node based on UCT and neural network prior.
        """
        return max(self.children, key=lambda child: child.uct_value_with_nn(exploration_param))

    def expand_with_nn(self, player, model):
        """
        Expands the tree using neural network policy probabilities.
        """
        if not self.untried_moves:
            return None

        _, policy_probs, value = predict_policy_and_value(model, self.board)
        sorted_moves = sorted(self.untried_moves, key=lambda m: policy_probs[m], reverse=True)

        move = sorted_moves.pop(0)
        new_board = copy.deepcopy(self.board)
        apply_player_action(new_board, move, player)

        child_node = MCTSNode_with_nn(new_board, parent=self, move=move, player=player, prior=policy_probs[move])
        child_node.value = value
        self.children.append(child_node)
        self.untried_moves.remove(move)

        return child_node

    def simulate_with_nn(self, model):
        """
        Uses the neural network to estimate win probability instead of random rollouts.
        """
        _, _, value = predict_policy_and_value(model, self.board)
        return value

    def backpropagation_with_nn(self, result):
        """
        Backpropagates results up the tree.
        """
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            current_node.wins += result
            current_node = current_node.parent


def forced_move(board, player):
    """
    Checks if there is a forced move (win or loss) for the current player.

    :param board: The current game board state.
    :param player: The player making the move.
    :return: The move that forces a win or loss or None if no such move exists.
    """
    valid_moves_list = valid_moves(board)

    # Check if any move results in a win for the current player
    for move in valid_moves_list:
        new_board = copy.deepcopy(board)
        apply_player_action(new_board, move, player)
        if check_end_state(new_board, player) == GameState.IS_WIN:
            return move  # Found a move that leads to a win

    # Check if any move results in a loss for the current player (opponent's win)
    opponent = PLAYER1 if player == PLAYER2 else PLAYER2
    for move in valid_moves_list:
        new_board = copy.deepcopy(board)
        apply_player_action(new_board, move, opponent)
        if check_end_state(new_board, opponent) == GameState.IS_WIN:
            return move  # Found a move that leads to a loss (avoid this move)

    # No forced win or loss move, return None
    return None


def generate_move_mcts_with_nn(board, player, save_state, iterations=500):
    """
    Generates a move using MCTS with neural network guidance.
    """
    forced_move_result = forced_move(board, player)
    if forced_move_result is not None:
        return forced_move_result, save_state

    model = load_model()

    root = MCTSNode_with_nn(board, player=player)

    for _ in range(iterations):
        node = root

        while node.children and not node.untried_moves:
            node = node.best_child_with_nn()

        if node.untried_moves:
            node = node.expand_with_nn(player, model)

        if node:
            result = node.simulate_with_nn(model)
            node.backpropagation_with_nn(result)

    return root.best_child_with_nn().move, save_state
