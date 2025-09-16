import os
import chess.pgn
import pickle
from tqdm import tqdm

def extract_positions_from_pgn(pgn_dir, min_move=20, max_positions=500_000, step=5):
    positions = set()  # use set to avoid duplicates
    collected = 0

    pbar = tqdm(total=max_positions, desc="Collecting positions")

    for file_name in os.listdir(pgn_dir):
        file_path = os.path.join(pgn_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()

                # Iterate moves
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)

                    # Skip openings → only start from move 20
                    if i < min_move:
                        continue

                    # Only take every 'step'-th move
                    if (i - min_move) % step != 0:
                        continue

                    fen = board.fen()

                    # Save FEN
                    if fen not in positions:
                        positions.add(fen)
                        collected += 1
                        pbar.update(1)

                        if collected >= max_positions:
                            pbar.close()
                            print(f"Collected {collected} positions (limit reached).")
                            return list(positions)

    pbar.close()
    print(f"Finished! Positions collected: {collected}")
    return list(positions)


def save_positions(positions, out_file):
    with open(out_file, "wb") as f:
        pickle.dump(positions, f)
    print(f"Saved {len(positions)} positions to {out_file}")


if __name__ == "__main__":
    positions = extract_positions_from_pgn("data/pgn", min_move=20, max_positions=500_000, step=5)
    save_positions(positions, "filtered_positions_from_database.pkl")
