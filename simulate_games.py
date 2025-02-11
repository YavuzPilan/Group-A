import os
import random
import numpy as np
import pandas as pd
import multiprocessing
import time

from game_utils import (
    initialize_game_state, apply_player_action, check_end_state, GameState,
    NO_PLAYER, PLAYER1, PLAYER2, valid_moves, pretty_print_board
)
from agents.agent_MCTS.MCTS import generate_move_mcts

# Path where the dataset will be saved
DATASET_PATH = os.path.join(os.getcwd(), "Data/dataset.csv")

# Lock to ensure safe writing to the CSV file in multiprocessing
csv_lock = multiprocessing.Lock()


def generate_policy_target(valid_moves, chosen_move, num_columns=7):
    """
    Creates a one-hot encoded policy vector representing the probability of each move.

    Args:
        valid_moves (list): List of available moves.
        chosen_move (int): Move selected by the agent.
        num_columns (int): Number of columns in the Connect 4 grid (default is 7).

    Returns:
        list: One-hot encoded policy vector with 1 at the chosen move index.
    """
    policy = [0] * num_columns
    if chosen_move in valid_moves:
        policy[chosen_move] = 1
    return policy


def select_move_with_temperature(node, temperature=1.0):
    """
    Selects a move based on visit counts of the MCTS node, using a temperature parameter.

    Args:
        node (MCTSNode): The root node of the MCTS search tree.
        temperature (float): Temperature parameter for probabilistic selection.
                            A value of 0 selects the most visited move deterministically.

    Returns:
        int: Chosen move index.

    Raises:
        ValueError: If the node is invalid or has no children.
    """
    if node is None or not hasattr(node, 'children'):
        raise ValueError("Invalid MCTS root node. Cannot perform temperature-based selection.")

    visit_counts = np.array([child.visits for child in node.children])

    if temperature == 0:
        return np.argmax(visit_counts)  # Always choose the most visited move

    # Apply temperature scaling to visit counts
    visit_probs = visit_counts ** (1 / temperature)
    visit_probs /= np.sum(visit_probs)  # Normalize to probabilities

    return np.random.choice(len(visit_counts), p=visit_probs)


def simulate_game(game_number, epsilon=0.1, temperature=1.0, random_start=False):
    """
    Simulates a Connect 4 game using MCTS and random exploration.

    Args:
        game_number (int): The index of the current game being simulated.
        epsilon (float): Probability of choosing a random move instead of MCTS.
        temperature (float): Temperature parameter for move selection.
        random_start (bool): Whether to start with a randomized board state.

    Returns:
        list: List of (board, policy, value) tuples.
    """
    process_name = multiprocessing.current_process().name

    print(f"Thread {process_name} Simulating game {game_number}...")  # Print current game number

    # Optionally start with a randomized board state
    if random_start:
        board = initialize_game_state()
        for _ in range(random.randint(0, 21)):  # Apply a random number of moves
            moves = valid_moves(board)
            if not moves:
                break
            random_move = random.choice(moves)
            apply_player_action(board, random_move, random.choice([PLAYER1, PLAYER2]))
    else:
        board = initialize_game_state()

    current_player = random.choice([PLAYER1, PLAYER2])
    game_data = []

    while True:
        moves = valid_moves(board)
        if not moves:
            break

        # Epsilon-greedy strategy: occasionally pick a random move
        if random.random() < epsilon:
            chosen_move = random.choice(moves)
        else:
            # Use MCTS to select the best move
            chosen_move, root_node = generate_move_mcts(board, current_player, None, iterations=1000)

            # Use temperature-based selection if enabled
            if temperature > 0 and root_node is not None:
                try:
                    chosen_move = select_move_with_temperature(root_node, temperature)
                except ValueError:
                    pass  # Fallback to the best move if selection fails

        # Generate policy vector for learning
        policy = generate_policy_target(moves, chosen_move, board.shape[1])
        game_data.append((board.copy(), policy, current_player))

        # Apply the move
        apply_player_action(board, chosen_move, current_player)
        state = check_end_state(board, current_player)

        # Assign rewards based on game outcome
        if state == GameState.IS_WIN:
            winner = current_player
            discount_factor = 0.95  # Future rewards are discounted
            for i in range(len(game_data)):
                state, policy, player = game_data[i]
                reward = 1 if player == winner else -1
                game_data[i] = (state, policy, reward * (discount_factor ** (len(game_data) - i - 1)))
            break
        elif state == GameState.IS_DRAW:
            game_data = [(data[0], data[1], 0) for data in game_data]  # Zero reward for draws
            break

        # Switch players
        current_player = PLAYER1 if current_player == PLAYER2 else PLAYER2
    print(f"Thread {process_name} Game {game_number} finished.")  # Print when game is done
    return game_data


def save_dataset(dataset_path, num_games=100, epsilon=0.1, temperature=1.0, random_start=False):
    """
    Runs multiple game simulations and saves results to a CSV file.

    Args:
        dataset_path (str): Path to save the dataset.
        num_games (int): Number of games to simulate.
        epsilon (float): Probability of random move selection.
        temperature (float): Temperature parameter for move selection.
        random_start (bool): Whether to start with a randomized board.

    Returns:
        None
    """
    dataset = []

    for game_number in range(1, num_games + 1):
        game_data = simulate_game(game_number, epsilon=epsilon, temperature=temperature, random_start=random_start)
        for board, policy, value in game_data:
            dataset.append({
                "Board": board.flatten().tolist(),
                "Policy": policy,
                "Value": value
            })

    df = pd.DataFrame(dataset)

    # Ensure only one process writes to the file at a time
    with csv_lock:
        df.to_csv(dataset_path, mode="a", index=False, header=not os.path.exists(dataset_path))


def worker(process_id, num_games, epsilon, temperature):
    """
    Worker function for parallel game simulations.

    Args:
        process_id (int): ID of the process.
        num_games (int): Number of games for this process.
        epsilon (float): Probability of random move selection.
        temperature (float): Temperature parameter for move selection.

    Returns:
        None
    """
    print(f"Process {process_id} started. Simulating {num_games} games.")
    save_dataset(DATASET_PATH, num_games, epsilon, temperature)
    print(f"Process {process_id} finished.")


if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()  # Use all CPU cores
    games_per_process = 200  # Number of games per process

    print(f"Number of CPUs {num_processes} in use.")
    start_time = time.perf_counter()

    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker, args=((i+1), games_per_process, 0.1, 1.0))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All processes completed!")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds")


