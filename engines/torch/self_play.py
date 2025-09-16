import chess
from chess import Board
import torch
import pickle
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from mcts import mcts_search
from predict import initialize_policy_model, initialize_value_model
import random
from multiprocessing import Pool, cpu_count
import subprocess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}
MATERIAL_THRESHOLD = 8
GameData = namedtuple("GameData", ["position", "pi", "z"])

# Global variables (one copy per process)
policy_model = None
value_model = None
int_to_move = None
device = None

def init_worker():
    """Initialize models once per worker."""
    global policy_model, value_model, int_to_move, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model, int_to_move = initialize_policy_model(device)
    value_model = initialize_value_model(device)

def evaluate_material(board: Board):
    white_score = sum(len(board.pieces(pt, True)) * val for pt, val in PIECE_VALUES.items())
    black_score = sum(len(board.pieces(pt, False)) * val for pt, val in PIECE_VALUES.items())
    return white_score - black_score

def play_from_position(start_fen, num_simulations=100):
    global policy_model, value_model, int_to_move, device

    board = Board(start_fen)
    game_data = []
    early_winner = None

    while not board.is_game_over():
        material_score = evaluate_material(board)
        if abs(material_score) >= MATERIAL_THRESHOLD:
            early_winner = 1 if material_score > 0 else -1
            break

        position = board.fen()
        _, _, pi = mcts_search(board, policy_model, value_model, device, int_to_move, num_simulations)

        non_zero_idx = np.nonzero(pi)[0]
        non_zero_probs = pi[non_zero_idx]
        non_zero_probs /= non_zero_probs.sum()

        game_data.append((position, pi))

        chosen_idx = np.random.choice(len(non_zero_idx), p=non_zero_probs)
        move_index = non_zero_idx[chosen_idx]
        move_uci = int_to_move[move_index]
        board.push_uci(move_uci)

    # Assign result
    if early_winner is not None:
        r = early_winner
    else:
        result = board.result()
        if result == "1-0":
            r = 1
        elif result == "0-1":
            r = -1
        else:
            r = 0

    #final_data = [GameData(position, pi, r) for position, pi, _ in game_data]

    final_data = []
    player = 1 if Board(start_fen).turn == chess.WHITE else -1
    for position, pi in game_data:
        z = r * player
        final_data.append(GameData(position, pi, z))
        player *= -1

    return final_data

def main_parallel():
    with open("data/pkl/filtered_positions_from_database.pkl", "rb") as f:
        start_positions = pickle.load(f)

    target_positions = 200_000
    self_play_data = []

    num_processes = min(cpu_count(), 4)
    print(f"Using {num_processes} processes")

    with Pool(processes=num_processes, initializer=init_worker) as pool:
        pbar = tqdm(total=target_positions, desc="Generating positions")

        while len(self_play_data) < target_positions:
            batch_size = num_processes * 2
            batch_positions = [random.choice(start_positions) for _ in range(batch_size)]

            results = pool.map(play_from_position, batch_positions)

            for game_data in results:
                self_play_data.extend(game_data)
                pbar.update(len(game_data))
                if len(self_play_data) >= target_positions:
                    break

        pbar.close()

    print(f"Collected {len(self_play_data)} positions.")

    with open("data/pkl/self-play_games_filtered_positions_2.pkl", "wb") as f:
        pickle.dump(self_play_data, f)
    print("Saved self-play positions.")


if __name__ == "__main__":
    main_parallel()
    subprocess.run(["python", "engines/torch/self-play_train.py"])
