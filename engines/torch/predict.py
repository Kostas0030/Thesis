from chess import Board, pgn, Move
import torch
from policy_network import PolicyNet
from value_network import ValueNet
import pickle
import numpy as np
from mcts import mcts_search

"""
Function to initialize the policy model

input: device
returns: policy_model, int_to_move
"""
def initialize_policy_model(device):
    with open("models/move_to_int", "rb") as file:
        move_to_int = pickle.load(file)

    policy_model = PolicyNet(num_classes=len(move_to_int))
    policy_model.load_state_dict(torch.load("/content/drive/MyDrive/Thesis/models/POLICY_MODEL_50EPOCHS.pth"))
    policy_model.to(device)
    policy_model.eval()

    int_to_move = {v: k for k, v in move_to_int.items()}

    return policy_model, int_to_move


"""
Function to initialize the value model

input: device
returns: value_model
"""
def initialize_value_model(device):
    value_model = ValueNet()
    value_model.load_state_dict(torch.load("/content/drive/MyDrive/Thesis/models/VALUE_MODEL_100EPOCHS.pth"))
    value_model.to(device)
    value_model.eval()

    return value_model


"""
Function to make mcts predictions

input: board state, model, device, int_to_move
returns: best_mcts_move
"""
def predict_move_with_mcts(board: Board, policy_model, value_model, device, int_to_move, simulations=200):
    return mcts_search(board, policy_model, value_model, device, int_to_move, num_simulations=simulations)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_model, int_to_move = initialize_policy_model(device)
    value_model = initialize_value_model(device)

    pgn_game = pgn.Game()
    node = pgn_game

    board = Board()

    first_move = "e2e3"
    board.push_uci("e2e3")
    node = node.add_variation(Move.from_uci(first_move))
    while not board.is_game_over():
        best_move, _, _ = predict_move_with_mcts(board, policy_model, value_model, device, int_to_move)
        board.push_uci(best_move)
        node = node.add_variation(Move.from_uci(best_move))
        
    print("=== PGN ===")
    print(pgn_game)

if __name__ == "__main__":

    main()
