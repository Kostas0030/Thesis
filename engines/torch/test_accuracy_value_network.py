import math
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from auxiliary_func import prepare_input_value_net, load_games_from_directory, extract_positions_from_games
from value_network import ValueNet
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained model
model = ValueNet().to(device)
model.load_state_dict(torch.load("models/TORCH_1EPOCHS_VALUE_NET.pth", map_location=device))
model.eval()

# Convert centipawn evaluation to [-1, 1] using Lichess formula
def centipawn_to_value(cp):
    win_percentage = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)  # 0 to 100
    prob = win_percentage / 100.0  # 0 to 1
    return 2 * prob - 1  # -1 to 1

def test_value_net_on_unseen_data():
    df = pd.read_csv("data/chessData.csv")

    df_sampled = df.sample(n=5000, random_state=42).copy() #5000 random samples

    df_sampled["Evaluation"] = pd.to_numeric(df_sampled["Evaluation"], errors="coerce") #convert Evaluation column to float
    df_sampled = df_sampled.dropna(subset=["Evaluation"])

    df_sampled["target"] = df_sampled["Evaluation"].apply(centipawn_to_value)

    X = [prepare_input_value_net(fen) for fen in df_sampled["FEN"]]
    X_tensor = torch.cat(X, dim=0).to(device)

    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()

    y_true = df_sampled["target"].values
    y_pred = preds

    return y_true, y_pred

def test_value_net_on_seen_data():
    print("Loading data...")
    games = load_games_from_directory("data/pgn", limit=20)
    print("Loading finished!")

    positions, results = extract_positions_from_games(games, max_positions=5_000)

    print("Preparing tensors for value network...")
    X = [prepare_input_value_net(fen) for fen in positions]
    X_tensor = torch.cat(X, dim=0)

    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()

    y_true = results
    y_pred = preds

    return y_true, y_pred

def main():
    y_true, y_pred = test_value_net_on_unseen_data()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Pearson correlation: {corr:.4f}")

    acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    print(f"Winner prediction accuracy: {acc:.4f}")


    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([-1, 1], [-1, 1], 'r--')  #perfect prediction line
    plt.xlabel("True evaluation [-1,1]")
    plt.ylabel("Network prediction [-1,1]")
    plt.title("Value Network Predictions vs True Evaluations")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()