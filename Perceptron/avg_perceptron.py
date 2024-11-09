import numpy as np
import pandas as pd
from helper import load_data, compute_error


def average_perceptron(X_train, y_train, epochs):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0
    average_weights = np.zeros(n_features)
    average_bias = 0

    for epoch in range(epochs):
        for idx, x in enumerate(X_train):
            if y_train[idx] * (np.dot(x, weights) + bias) <= 0:
                weights += y_train[idx] * x
                bias += y_train[idx]
            average_weights += weights
            average_bias += bias

    return average_weights / (epochs * n_samples), average_bias / (epochs * n_samples)


def predict(X, weights, bias):
    return np.sign(np.dot(X, weights) + bias)


train_file = "bank-note/train.csv"
test_file = "bank-note/test.csv"

epochs = 10

X_train, y_train, X_test, y_test = load_data(train_file, test_file)

average_weights, average_bias = average_perceptron(X_train, y_train, epochs)

y_pred = predict(X_test, average_weights, average_bias)
test_error = compute_error(y_test, y_pred)

print("Learned weight vector (averaged):", average_weights)
print("Learned bias (averaged):", average_bias)
print("Average prediction error on the test dataset:", test_error)
