import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from predict import initialize_policy_model, initialize_value_model
from mcts import mcts_search

def init_worker(policy_new_path, policy_old_path, value_new_path, value_old_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model_new, int_to_move = initialize_policy_model(device)
    policy_model_old, _ = initialize_policy_model(device)
    value_model_new = initialize_value_model(device)
    value_model_old = initialize_value_model(device)

    policy_model_new.load_state_dict(torch.load(policy_new_path, map_location=device))
    policy_model_old.load_state_dict(torch.load(policy_old_path, map_location=device))
    value_model_new.load_state_dict(torch.load(value_new_path, map_location=device))
    value_model_old.load_state_dict(torch.load(value_old_path, map_location=device))

    policy_model_new.eval()
    policy_model_old.eval()
    value_model_new.eval()
    value_model_old.eval()

    return policy_model_new, policy_model_old, value_model_new, value_model_old, int_to_move, device

def select_move_mcts(board, policy_model, value_model, int_to_move, device, num_simulations=200):
    _, _, pi = mcts_search(board, policy_model, value_model, device, int_to_move, num_simulations)

    legal_moves = list(board.legal_moves)
    move_uci_to_index = {int_to_move[i]: i for i in range(len(pi))}

    legal_indices, legal_probs = [], []
    for move in legal_moves:
        uci = move.uci()
        if uci in move_uci_to_index:
            idx = move_uci_to_index[uci]
            legal_indices.append(idx)
            legal_probs.append(pi[idx])

    legal_probs = np.array(legal_probs)
    legal_probs /= legal_probs.sum()
    chosen_idx = np.random.choice(len(legal_indices), p=legal_probs)
    return chess.Move.from_uci(int_to_move[legal_indices[chosen_idx]])

def play_game_task(args):
    i, _, num_simulations, models = args
    policy_new, policy_old, value_new, value_old, int_to_move, device = models

    board = chess.Board()
    game = chess.pgn.Game()
    
    # Alternate colors: new model White if even, Black if odd
    if i % 2 == 0:
        white_policy, white_value = policy_new, value_new
        black_policy, black_value = policy_old, value_old
        game.headers["White"] = "Model_New"
        game.headers["Black"] = "Model_Old"
    else:
        white_policy, white_value = policy_old, value_old
        black_policy, black_value = policy_new, value_new
        game.headers["White"] = "Model_Old"
        game.headers["Black"] = "Model_New"

    node = game
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = select_move_mcts(board, white_policy, white_value, int_to_move, device, num_simulations)
        else:
            move = select_move_mcts(board, black_policy, black_value, int_to_move, device, num_simulations)
        board.push(move)
        node = node.add_variation(move)

    result = board.result()
    game.headers["Result"] = result

    # Outcome from new model's perspective
    if i % 2 == 0:  # new model was White
        outcome = 1 if result == "1-0" else (-1 if result == "0-1" else 0)
    else:           # new model was Black
        outcome = 1 if result == "0-1" else (-1 if result == "1-0" else 0)

    return outcome, game

def evaluate_models_parallel(policy_new_path, policy_old_path, value_new_path, value_old_path, n_games=10, mode="mcts", num_simulations=200):
    models = init_worker(policy_new_path, policy_old_path, value_new_path, value_old_path)
    num_processes = min(cpu_count(), 4)
    print(f"Using {num_processes} processes")

    args = [(i, mode, num_simulations, models) for i in range(n_games)]

    games = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(play_game_task, args), total=n_games, desc="Evaluating"))

    scores = {1: 0, 0: 0, -1: 0}
    for outcome, game in results:
        scores[outcome] += 1
        games.append(game)

    print("\n=== Final Results ===")
    print(f"Model new wins: {scores[1]}")
    print(f"Draws: {scores[0]}")
    print(f"Model old wins: {scores[-1]}")

    # Print PGNs and colors
    for i, game in enumerate(games, 1):
        print(f"\n=== Game {i} ===")
        print(f"White: {game.headers['White']}, Black: {game.headers['Black']}, Result: {game.headers['Result']}")
        print(game)

def main():
    policy_new_path = "models/POLICY_MODEL_50EPOCHS_V1.pth"
    policy_old_path = "models/POLICY_MODEL_50EPOCHS.pth"
    value_new_path = "models/VALUE_MODEL_100EPOCHS_V1.pth"
    value_old_path = "models/VALUE_MODEL_100EPOCHS.pth"

    evaluate_models_parallel(policy_new_path, policy_old_path, value_new_path, value_old_path, n_games=700, mode="mcts", num_simulations=200)

if __name__ == "__main__":
    main()
