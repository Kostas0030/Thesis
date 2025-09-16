import pickle
from chess import Board, pgn
from collections import namedtuple


GameData = namedtuple("GameData", ["position", "pi", "z"])


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

with open("data/pkl/self-play_games.pkl", "rb") as f:
    all_game_data = pickle.load(f)

games = []
current_game = []

for data in all_game_data:
    if data.position == STARTING_FEN and current_game:
        games.append(current_game)
        current_game = []

    current_game.append(data)

if current_game:
    games.append(current_game)

print(f"Loaded {len(games)} games\n")

def generate_pgn_from_game(game_data):
    game = pgn.Game()
    node = game

    for i in range(len(game_data) - 1):
        board_current = Board(game_data[i].position)
        board_next = Board(game_data[i+1].position)

        move_found = False
        for move in board_current.legal_moves:
            board_copy = board_current.copy()
            board_copy.push(move)
            if board_copy.fen() == board_next.fen():
                node = node.add_variation(move)
                move_found = True
                break

        if not move_found:
            print(f"Warning: No legal move from {board_current.fen()} to {board_next.fen()}")

    last_result = game_data[-1].z
    if last_result == 1:
        game.headers["Result"] = "1-0"
    elif last_result == -1:
        game.headers["Result"] = "0-1"
    else:
        game.headers["Result"] = "1/2-1/2"

    return game

for idx, game_data in enumerate(games, start=1):
    print(f"\n=== Game {idx} ===")
    pgn_game = generate_pgn_from_game(game_data)
    print(pgn_game)
