from chess import Board
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from auxiliary_func import prepare_input_policy_net, prepare_input_value_net
from predict import initialize_policy_model, initialize_value_model
from collections import namedtuple


GameData = namedtuple("GameData", ["position", "pi", "z"])

class SelfPlayDataset(Dataset):
    def __init__(self, pickle_file, policy_length):
        with open(pickle_file, "rb") as f:
            self.data = pickle.load(f)  # list of GameData(position, pi, z)
        self.move_count = policy_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        position, pi, z = self.data[idx]

        board = Board(position)
        x_policy = prepare_input_policy_net(board).squeeze(0)
        x_value = prepare_input_value_net(board).squeeze(0)

        pi = torch.tensor(pi, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(0)

        return x_policy, x_value, pi, z


def train_policy(policy_model, dataset, device, epochs=10, batch_size=128, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.KLDivLoss(reduction="batchmean")  # good for distributions
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)

    policy_model.to(device)
    policy_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x, _, target_pi, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, target_pi = x.to(device), target_pi.to(device)

            optimizer.zero_grad()
            pred_logits = policy_model(x)  # shape: [batch, policy_length]
            pred_log_probs = torch.log_softmax(pred_logits, dim=1)

            loss = criterion(pred_log_probs, target_pi)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    return policy_model

def train_value(value_model, dataset, device, epochs=10, batch_size=128, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(value_model.parameters(), lr=lr)

    value_model.to(device)
    value_model.train()

    for epoch in range(epochs):
        total_loss = 0
        for _, x, _, target_z in tqdm(dataloader, desc=f"Value Epoch {epoch+1}/{epochs}"):
            x, target_z = x.to(device), target_z.to(device)

            optimizer.zero_grad()
            pred_value = value_model(x)
            loss = criterion(pred_value, target_z)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Value Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")

    return value_model



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_model, int_to_move = initialize_policy_model(device)
    value_model = initialize_value_model(device)

    dataset = SelfPlayDataset("data/pkl/self-play_games_filtered_positions_2.pkl", len(int_to_move))

    trained_policy = train_policy(policy_model, dataset, device, epochs=100, batch_size=128, lr=1e-4)
    torch.save(trained_policy.state_dict(), "models/POLICY_MODEL_50EPOCHS_V1.pth")
    print("Saved policy model")

    trained_value = train_value(value_model, dataset, device, epochs=100, batch_size=128, lr=1e-4)
    torch.save(trained_value.state_dict(), "models/VALUE_MODEL_100EPOCHS_V1.pth")
    print("Saved value model")

if __name__ == "__main__":
    main()
