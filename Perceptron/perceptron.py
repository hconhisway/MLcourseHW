import numpy as np
from helper import load_data, compute_error


def perceptron(X_train, y_train, epochs):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        for idx, x in enumerate(X_train):
            if y_train[idx] * (np.dot(x, weights) + bias) <= 0:
                weights += y_train[idx] * x
                bias += y_train[idx]

    return weights, bias


def predict(X, weights, bias):
    return np.sign(np.dot(X, weights) + bias)


# File paths (update these to the actual file paths)
train_file = "bank-note/train.csv"
test_file = "bank-note/test.csv"

# Parameters
epochs = 10

# Load data
X_train, y_train, X_test, y_test = load_data(train_file, test_file)

# Train perceptron
weights, bias = perceptron(X_train, y_train, epochs)

# Evaluate on test set
y_pred = predict(X_test, weights, bias)
test_error = compute_error(y_test, y_pred)

# Results
print("Learned weight vector:", weights)
print("Learned bias:", bias)
print("Average prediction error on the test dataset:", test_error)
