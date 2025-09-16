import chess
import math
import numpy as np
from auxiliary_func import prepare_input_policy_net, prepare_input_value_net
import torch
import random

class Node:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0  # visit count
        self.W = 0  # total value
        self.P = 0  # prior from model
        self.Q = 0  # mean value (exploitation)

    def is_expanded(self):
        return len(self.children) > 0

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def mcts_search(root_board, policy_model, value_model, device, int_to_move, num_simulations, c_puct=1.5):
    root = Node(root_board)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # === SELECTION ===
        while node.is_expanded():
            max_ucb = -float("inf")
            best_child = None
            for move, child in node.children.items():
                ucb = child.Q + c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_child = child
            node = best_child
            search_path.append(node)

        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0
        else:
            # === EXPANSION & EVALUATION ===
            # Get policy priors for this leaf
            input_tensor = prepare_input_policy_net(node.board).to(device)
            with torch.no_grad():
                logits = policy_model(input_tensor).squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            # Get value from value network
            input_tensor_val = prepare_input_value_net(node.board).to(device)
            with torch.no_grad():
                value = value_model(input_tensor_val).item()

            legal_moves = list(node.board.legal_moves)
            move_uci_to_index = {int_to_move[i]: i for i in range(len(probs))}
            total_prob = 0.0
            for move in legal_moves:
                uci = move.uci()
                if uci in move_uci_to_index:
                    idx = move_uci_to_index[uci]
                    prob = probs[idx]
                    child_board = node.board.copy()
                    child_board.push(move)
                    child_node = Node(child_board, parent=node, move=move)
                    child_node.P = prob
                    node.children[move] = child_node
                    total_prob += prob

            # Normalize priors
            if node.children:
                for child in node.children.values():
                    child.P /= total_prob + 1e-8

        # === BACKPROPAGATION ===
        for back_node in reversed(search_path):
            back_node.N += 1
            back_node.W += value
            back_node.Q = back_node.W / back_node.N
            value = -value  # Switch perspective

    # === PICK THE MOST VISITED MOVE ===
    if not root.children:
        return None, root.Q, None
    
    # Build visit count distribution π
    legal_moves = list(root.children.keys())
    visits = np.array([child.N for child in root.children.values()], dtype=np.float32)
    pi = visits / np.sum(visits)

    # Map moves to policy vector
    policy = np.zeros(len(int_to_move), dtype=np.float32)
    for move, child in root.children.items():
        uci = move.uci()
        if uci in int_to_move.values():
            idx = list(int_to_move.keys())[list(int_to_move.values()).index(uci)]
            policy[idx] = pi[legal_moves.index(move)]

    #start_player = root.board.turn
    #if start_player == True:
        #print("White to play!")
    #else:
        #print("Black to play!")
    #print(root.board)
    #for move, child in root.children.items():
        #print(f"Move: {move.uci()}, Visits: {child.N}, Q: {child.Q:.4f}")

    best_move = max(root.children.items(), key=lambda item: item[1].N)[0].uci()
    #print("Best move: ", best_move)
    return best_move, root.Q, policy


def evaluate_leaf(board: chess.Board, model, device, int_to_move):
    """
    Simplified evaluation using only the policy network:
    if it's White's turn: 1.0 - top move probability
    if it's Black's turn: -1.0 + top move probability
    (this is just a placeholder; ideally use a value net)
    """
    input_tensor = prepare_input_policy_net(board).to(device)
    with torch.no_grad():
        logits = model(input_tensor).squeeze(0)
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    move_uci_to_index = {int_to_move[i]: i for i in range(len(probs))}
    top_prob = 0.0
    for move in legal_moves:
        uci = move.uci()
        if uci in move_uci_to_index:
            idx = move_uci_to_index[uci]
            top_prob = max(top_prob, probs[idx])
    return top_prob if board.turn == chess.WHITE else -top_prob



def evaluate_leaf_with_policy_rollout(board: chess.Board, model, device, int_to_move, depth_limit=20):
    """
    Evaluate a leaf node using a rollout guided by the policy network.
    Plays moves until game over or depth_limit is reached.
    Returns result from the perspective of the leaf's current player.
    """
    start_player = board.turn  # True for white, False for black
    sim_board = board.copy()

    for depth in range(depth_limit):
        if sim_board.is_game_over():
            break

        # Get move probabilities from the policy net
        input_tensor = prepare_input_policy_net(sim_board).to(device)
        with torch.no_grad():
            logits = model(input_tensor).squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Map legal moves to their probabilities
        legal_moves = list(sim_board.legal_moves)
        move_uci_to_index = {int_to_move[i]: i for i in range(len(probs))}
        move_probs = []
        for move in legal_moves:
            uci = move.uci()
            if uci in move_uci_to_index:
                idx = move_uci_to_index[uci]
                move_probs.append(probs[idx])
            else:
                move_probs.append(0.0)

        # Normalize (avoid all-zero probabilities)
        total_prob = sum(move_probs)
        if total_prob > 0:
            move_probs = [p / total_prob for p in move_probs]
        else:
            move_probs = [1 / len(move_probs)] * len(move_probs)  # fallback to uniform

        # Sample move according to policy probabilities
        chosen_move = random.choices(legal_moves, weights=move_probs, k=1)[0]
        sim_board.push(chosen_move)

    # Final evaluation
    if sim_board.is_game_over():
        result = sim_board.result()  # e.g., "1-0", "0-1", "1/2-1/2"
        if result == "1-0":
            return 1 if start_player == chess.WHITE else -1
        elif result == "0-1":
            return -1 if start_player == chess.WHITE else 1
        else:
            return 0
    else:
        # If we reached depth limit without ending, use simple heuristic: material difference
        return simple_material_eval(sim_board, start_player)


def simple_material_eval(board: chess.Board, start_player: bool):
    """
    Quick material evaluation for non-terminal positions.
    Returns value from start_player's perspective.
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9
    }
    score = 0
    for piece_type, value in piece_values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # Normalize to range -1 to 1
    score = max(-15, min(15, score)) / 15
    return score if start_player == chess.WHITE else -score