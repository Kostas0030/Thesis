import chess
import chess.pgn
import torch
import numpy as np
import pickle
from policy_network import PolicyNet
from auxiliary_func import prepare_input_policy_net

"""
Function to initialize the policy model

input: device
returns: policy_model, int_to_move
"""
def initialize_policy_model(device):
    with open("models/move_to_int", "rb") as file:
        move_to_int = pickle.load(file)

    policy_model = PolicyNet(num_classes=len(move_to_int))
    policy_model.load_state_dict(torch.load("models/POLICY_MODEL_50EPOCHS.pth"))
    policy_model.to(device)
    policy_model.eval()

    int_to_move = {v: k for k, v in move_to_int.items()}

    return policy_model, int_to_move

def select_move_policy(board, policy_model, int_to_move, device):
    """Selects a move directly from the policy network's probabilities."""
    x = prepare_input_policy_net(board).to(device)
    with torch.no_grad():
        logits = policy_model(x).squeeze(0)
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    move_uci_to_index = {int_to_move[i]: i for i in range(len(probs))}

    legal_indices, legal_probs = [], []
    for move in legal_moves:
        uci = move.uci()
        if uci in move_uci_to_index:
            idx = move_uci_to_index[uci]
            legal_indices.append(idx)
            legal_probs.append(probs[idx])

    legal_probs = np.array(legal_probs)
    legal_probs /= legal_probs.sum()

    chosen_idx = np.random.choice(len(legal_indices), p=legal_probs)
    return chess.Move.from_uci(int_to_move[legal_indices[chosen_idx]])


def play_self_game(policy_model, int_to_move, device, max_moves=200):
    """Play a full self-play game with the policy net."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "PolicyNet"
    game.headers["Black"] = "PolicyNet"
    node = game

    move_num = 1
    while not board.is_game_over() and move_num <= max_moves:
        move = select_move_policy(board, policy_model, int_to_move, device)
        board.push(move)
        node = node.add_variation(move)
        move_num += 1

    game.headers["Result"] = board.result()
    return game


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model, int_to_move = initialize_policy_model(device)

    game = play_self_game(policy_model, int_to_move, device)

    print(game)


if __name__ == "__main__":
    main()
