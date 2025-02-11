import threading
import time
import csv
from typing import Callable

from agents.agent_MCTS.MCTS import generate_move_mcts
from agents.agent_Minimax.Minimax import generate_move_minimax
from agents.agent_random.random import generate_move_random
from game_utils import PLAYER1, PLAYER2, GameState, MoveStatus, GenMove
from game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, \
    check_move_status
from agents.agent_MCTS_with_nn.MCTS_with_nn import generate_move_mcts_with_nn
from plot.plots import get_relative_path


def human_vs_agent(
        generate_move_1: GenMove,
        generate_move_2: GenMove,
        player_1: str = "Neural Network",
        player_2: str = "Other agent",
        args_1: tuple = (),
        args_2: tuple = (),
        init_1: Callable = lambda board, player: None,
        init_2: Callable = lambda board, player: None,
):
    players = (PLAYER1, PLAYER2)
    wins_nn, wins_other, draws = 0, 0, 0

    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        # Find out which player is the NN agent
        nn_player = players[player_names.index("Neural Network")]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(players, player_names, gen_moves, gen_args):
                t0 = time.time()
                # print(pretty_print_board(board))
                # print(f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}')
                action, saved_state[player] = gen_move(
                    board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                    player, saved_state[player], *args
                )
                # print(f'Move time: {time.time() - t0:.3f}s')

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    # print(f'Move {action} is invalid: {move_status.value}')
                    # print(f'{player_name} lost by making an illegal move.')
                    if player == nn_player:
                        wins_other += 1  # NN lost
                    else:
                        wins_nn += 1  # NN won
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    # print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print(f'{player_name} draw against {player_2}')
                        print(pretty_print_board(board))
                        draws += 1
                    else:
                        if player == nn_player:
                            print(pretty_print_board(board))
                            print(f'{player_1} won against {player_2}')
                            wins_nn += 1  # NN won
                        else:
                            print(pretty_print_board(board))
                            print(f'{player_2} won against {player_1}')
                            wins_other += 1  # NN lost
                    playing = False
                    break

    return wins_nn, wins_other, draws


def save_results_to_csv(filename, methods, win_rates):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Win Rate (%)"])
        for method, rate in zip(methods, win_rates):
            writer.writerow([method, rate])


def run_games(agent, name, results, total_games=10):
    wins_nn, wins_other, draws = 0, 0, 0
    for _ in range(total_games):
        w1, w2, d = human_vs_agent(generate_move_1=generate_move_mcts_with_nn, generate_move_2=agent, player_2=name)
        wins_nn += w1
        wins_other += w2
        draws += d

    win_rate = (wins_nn / (wins_nn + wins_other + draws)) * 100 if (wins_nn + wins_other + draws) > 0 else 0
    results[name] = win_rate
    print(f"Neural Network vs. {name}\nTotal Games: {(wins_nn + wins_other + draws)}\nGames Won: {wins_nn}")


if __name__ == "__main__":
    total_games = 1  # Adjust as needed
    methods = ["Random Bot", "Minimax", "MCTS"]
    agents = [generate_move_random, generate_move_minimax, generate_move_mcts]

    results = {}
    threads = []

    for agent, name in zip(agents, methods):
        thread = threading.Thread(target=run_games, args=(agent, name, results, total_games))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to finish

    # Save results to CSV and plot
    csv_filename = get_relative_path("..", "Data", "win_rates.csv")
    save_results_to_csv(csv_filename, methods, [results[m] for m in methods])

    print("Win rate comparison saved.")
