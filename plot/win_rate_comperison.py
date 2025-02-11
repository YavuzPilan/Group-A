import threading
import time
import csv
from typing import Callable

from agents.agent_MCTS.MCTS import generate_move_mcts
from agents.agent_Minimax.Minimax import generate_move_minimax
from agents.agent_random.random import generate_move_random
from game_utils import PLAYER1, PLAYER2, GameState, MoveStatus, GenMove, PLAYER1_PRINT, PLAYER2_PRINT
from game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, \
    check_move_status
from agents.agent_MCTS_with_nn.MCTS_with_nn import generate_move_mcts_with_nn
from plot.plots import get_relative_path

lock = threading.Lock()  # Lock für Thread-Sicherheit


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

        nn_player = players[player_names.index("Neural Network")]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(players, player_names, gen_moves, gen_args):
                print(pretty_print_board(board))
                print(f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}')
                action, saved_state[player] = gen_move(
                    board.copy(),
                    player, saved_state[player], *args
                )

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    if player == nn_player:
                        wins_other += 1
                    else:
                        wins_nn += 1
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    if end_state == GameState.IS_DRAW:
                        draws += 1
                    else:
                        if player == nn_player:
                            wins_nn += 1
                        else:
                            wins_other += 1
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

    with lock:  # Thread-Safe Zugriff auf das Dictionary
        results[name] = win_rate

    print(f"Neural Network vs. {name}\nTotal Games: {wins_nn + wins_other + draws}\nGames Won: {wins_nn}")


if __name__ == "__main__":
    total_games = 5
    methods = ["Random Bot", "Minimax", "MCTS"]
    agents = [generate_move_random, generate_move_minimax, generate_move_mcts]

    results = {}
    threads = []

    for agent, name in zip(agents, methods):
        thread = threading.Thread(target=run_games, args=(agent, name, results, total_games))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Final Results:", results)  # Debugging, um sicherzustellen, dass alles gesetzt wurde

    # Sicherstellen, dass alle Methoden in den Ergebnissen sind
    win_rates = [results.get(m, 0) for m in methods]  # Falls ein Wert fehlt, wird 0 gesetzt

    csv_filename = get_relative_path("..", "Data", "win_rates.csv")
    save_results_to_csv(csv_filename, methods, win_rates)

    print("Win rate comparison saved.")
