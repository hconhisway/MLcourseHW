import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, activation, init_type):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

        for layer in self.layers:
            if init_type == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

train_X, train_y = load_data("bank-note/train.csv")
test_X, test_y = load_data("bank-note/test.csv")

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activation_functions = {"tanh": nn.Tanh(), "relu": nn.ReLU()}
initializations = {"tanh": "xavier", "relu": "he"}

results = {}

for activation_name, activation_func in activation_functions.items():
    for depth in depths:
        for width in widths:
            model = NeuralNetwork(input_dim=train_X.shape[1], hidden_dim=width, output_dim=1, depth=depth, activation=activation_func, init_type=initializations[activation_name])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            epochs = 30
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_X).squeeze()
                loss = criterion(outputs, train_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                train_predictions = model(train_X).squeeze()
                train_error = criterion(train_predictions, train_y).item()

                test_predictions = model(test_X).squeeze()
                test_error = criterion(test_predictions, test_y).item()

            results[(activation_name, depth, width)] = (train_error, test_error)

print("Activation, Depth, Width -> Train Error, Test Error")
for (activation, depth, width), (train_err, test_err) in results.items():
    print(f"{activation}, {depth}, {width} -> {train_err:.4f}, {test_err:.4f}")
