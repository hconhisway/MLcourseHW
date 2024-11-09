import numpy as np
import pandas as pd
from helper import load_data, compute_error


def voted_perceptron(X_train, y_train, epochs):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0
    weight_vectors = []
    counts = []

    # Initialize counts for the first weight vector
    m = 0
    for epoch in range(epochs):
        for idx, x in enumerate(X_train):
            if y_train[idx] * (np.dot(x, weights) + bias) <= 0:
                # Store the current weights and their count
                weight_vectors.append((weights.copy(), bias))
                counts.append(m)
                # Update weights and reset count
                weights += y_train[idx] * x
                bias += y_train[idx]
                m = 1
            else:
                m += 1

    # Append the last weight vector
    weight_vectors.append((weights.copy(), bias))
    counts.append(m)
    return weight_vectors, counts

def predict_voted(X, weight_vectors, counts):
    predictions = np.zeros(X.shape[0])
    for (weights, bias), count in zip(weight_vectors, counts):
        predictions += count * np.sign(np.dot(X, weights) + bias)
    return np.sign(predictions)


train_file = "bank-note/train.csv"
test_file = "bank-note/test.csv"

epochs = 10

X_train, y_train, X_test, y_test = load_data(train_file, test_file)

weight_vectors, counts = voted_perceptron(X_train, y_train, epochs)

y_pred = predict_voted(X_test, weight_vectors, counts)
test_error = compute_error(y_test, y_pred)

print("List of distinct weight vectors and their counts:")
for i, ((weights, bias), count) in enumerate(zip(weight_vectors, counts)):
    print(f"Vector {i+1}: Weights: {weights}, Bias: {bias}, Count: {count}")

print("\nAverage prediction error on the test dataset:", test_error)