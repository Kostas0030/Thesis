import torch
import chess
from value_network import ValueNet
from auxiliary_func import prepare_input_value_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ValueNet().to(device)
model.load_state_dict(torch.load("models/VALUE_MODEL_100EPOCHS.pth", map_location=device))
model.eval()

fen = "8/8/7n/2k5/8/p5K1/P7/8 b - - 15 96"

X_test = prepare_input_value_net(chess.Board(fen)).to(device)

with torch.no_grad():
    predicted_value = model(X_test)
    print("Predicted value:", predicted_value.item())
