import chess
import chess.pgn
from chess import Board
import torch
import numpy as np
import pickle
from collections import namedtuple
from mcts import mcts_search
from predict import initialize_policy_model, initialize_value_model

# Constants
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

def evaluate_material(board: Board):
    white_score = sum(len(board.pieces(pt, True)) * val for pt, val in PIECE_VALUES.items())
    black_score = sum(len(board.pieces(pt, False)) * val for pt, val in PIECE_VALUES.items())
    return white_score - black_score

def play_one_debug(start_fen, policy_model, value_model, device, int_to_move, num_simulations=50):
    """Play ONE game from a given start position and print debug info."""
    board = Board(start_fen)
    game_data = []
    early_winner = None

    move_num = 1
    while not board.is_game_over():
        print(f"\n=== Move {move_num} ===")
        print(board)

        # Early termination
        material_score = evaluate_material(board)
        if abs(material_score) >= MATERIAL_THRESHOLD:
            early_winner = 1 if material_score > 0 else -1
            print(f"Early termination (material diff {material_score}), winner = {early_winner}")
            break

        # MCTS
        _, _, pi = mcts_search(board, policy_model, value_model, device, int_to_move, num_simulations)

        # Debug: top moves
        non_zero_idx = np.nonzero(pi)[0]
        non_zero_probs = pi[non_zero_idx]
        non_zero_probs /= non_zero_probs.sum()

        print("Top candidate moves:")
        sorted_idx = np.argsort(-non_zero_probs)
        for i in sorted_idx[:5]:
            move_uci = int_to_move[non_zero_idx[i]]
            print(f"  {move_uci}: {non_zero_probs[i]:.3f}")

        # Save (s, π, None)
        game_data.append((board.fen(), pi.copy()))

        # Pick a move
        chosen_idx = np.random.choice(len(non_zero_idx), p=non_zero_probs)
        move_index = non_zero_idx[chosen_idx]
        move_uci = int_to_move[move_index]
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        print(f"Chosen move: {move_uci}")

        move_num += 1

    # Game result
    if early_winner is not None:
        r = early_winner
        result = "1-0" if r == 1 else "0-1"
    else:
        result = board.result()
        if result == "1-0":
            r = 1
        elif result == "0-1":
            r = -1
        else:
            r = 0
        print(f"\nGame over. Final result: {result}  =>  z = {r}")

    # Apply result
    final_data = []
    player = 1 if Board(start_fen).turn == chess.WHITE else -1
    for position, pi in game_data:
        z = r * player
        final_data.append(GameData(position, pi, z))
        player *= -1

    # Build PGN cleanly from the final board
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "Debug_White"
    game.headers["Black"] = "Debug_Black"
    game.headers["Result"] = result

    return final_data, game

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model, int_to_move = initialize_policy_model(device)
    value_model = initialize_value_model(device)

    # Load a starting position from your database
    with open("data/pkl/filtered_positions_from_database.pkl", "rb") as f:
        start_positions = pickle.load(f)

    start_fen = start_positions[0]
    print(f"Starting from FEN: {start_fen}")

    game_data, game = play_one_debug(start_fen, policy_model, value_model, device, int_to_move)

    print("\n=== Final collected samples ===")
    for i, gd in enumerate(game_data, 1):
        board = chess.Board(gd.position)
        print(f"\n--- Position {i} ---")
        print(board.fen())
        print(f"z = {gd.z}")
        non_zero_idx = np.nonzero(gd.pi)[0]
        non_zero_probs = gd.pi[non_zero_idx]
        non_zero_probs /= non_zero_probs.sum()
        sorted_idx = np.argsort(-non_zero_probs)
        print("Policy (top 5 moves):")
        for j in sorted_idx[:5]:
            move_uci = int_to_move[non_zero_idx[j]]
            print(f"  {move_uci}: {non_zero_probs[j]:.3f}")

    print("\n=== PGN of the game ===")
    print(game)

if __name__ == "__main__":
    main()
