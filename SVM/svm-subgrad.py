import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('bank-note/train.csv', header=None)
test_data = pd.read_csv('bank-note/test.csv', header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)
test_data.iloc[:, -1] = test_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

T = 100  # Maximum epochs
C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_0 = 0.1  # Initial learning rate
a = 1  # Hyperparameter for learning rate decay


def svm_sgd(X, y, C, gamma_0, a, T):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    # Track losses
    losses = []

    for epoch in range(T):
        # Shuffle the data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        epoch_loss = 0
        for t, (xi, yi) in enumerate(zip(X, y), 1):
            gamma_t = gamma_0 / (1 + gamma_0 * t / a)  # Learning rate schedule
            if yi * (np.dot(w, xi) + b) < 1:  # Hinge loss condition
                w = (1 - gamma_t) * w + gamma_t * C * yi * xi
                b += gamma_t * C * yi
                epoch_loss += max(0, 1 - yi * (np.dot(w, xi) + b))
            else:
                w = (1 - gamma_t) * w
        losses.append(epoch_loss / n_samples)

    return w, b, losses


def svm_sgd_with_schedule_b(X, y, C, gamma_0, T):
    """
    SVM with Stochastic Sub-Gradient Descent using the learning rate schedule:
    gamma_t = gamma_0 / (1 + t).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias

    # Track losses
    losses = []

    for epoch in range(T):
        # Shuffle the data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        epoch_loss = 0
        for t, (xi, yi) in enumerate(zip(X, y), 1):
            gamma_t = gamma_0 / (1 + t)  # Updated learning rate schedule
            if yi * (np.dot(w, xi) + b) < 1:  # Hinge loss condition
                w = (1 - gamma_t) * w + gamma_t * C * yi * xi
                b += gamma_t * C * yi
                epoch_loss += max(0, 1 - yi * (np.dot(w, xi) + b))
            else:
                w = (1 - gamma_t) * w
        losses.append(epoch_loss / n_samples)

    return w, b, losses


# Train and evaluate for different C values
results = {}

for C in C_values:
    w, b, losses = svm_sgd(X_train, y_train, C, gamma_0, a, T)

    # Predict on training and test data
    y_train_pred = np.sign(X_train @ w + b)
    y_test_pred = np.sign(X_test @ w + b)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    results[C] = {
        "weights": w,
        "bias": b,
        "train_error": train_error,
        "test_error": test_error,
        "losses": losses
    }
results_b = {}

for C in C_values:
    w, b, losses = svm_sgd_with_schedule_b(X_train, y_train, C, gamma_0, T)

    # Predict on training and test data
    y_train_pred = np.sign(X_train @ w + b)
    y_test_pred = np.sign(X_test @ w + b)

    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    results_b[C] = {
        "weights": w,
        "bias": b,
        "train_error": train_error,
        "test_error": test_error,
        "losses": losses
    }

comparison_results = {}

for C in C_values:
    weights = results[C]['weights']
    weights_b = results_b[C]['weights']
    weights_diff = np.linalg.norm(weights - weights_b)
    bias_diff = abs(results[C]['bias'] - results_b[C]['bias'])
    train_error_diff = results[C]['train_error'] - results_b[C]['train_error']
    test_error_diff = results[C]['test_error'] - results_b[C]['test_error']

    comparison_results[C] = {
        "weights": weights,  # Save the actual weights
        "weights_b": weights_b,  # Save the comparison weights
        "weights_diff": weights_diff,  # Save the weight difference
        "bias_diff": bias_diff,
        "train_error_diff": train_error_diff,
        "test_error_diff": test_error_diff
    }



# Display results
for C, result in results.items():
    print(f"C = {C}")
    print(f"Train Error: {result['train_error']:.4f}")
    print(f"Test Error: {result['test_error']:.4f}")
    print("-" * 30)

print("-" * 50)

for C, result in results_b.items():
    print(f"C = {C}")
    print(f"Train Error: {result['train_error']:.4f}")
    print(f"Test Error: {result['test_error']:.4f}")
    print("-" * 30)

# Display comparison results
for C, comp_result in comparison_results.items():
    print(f"C = {C}")
    print(f"weights (results): {comp_result['weights']}")
    print(f"weights (results_b): {comp_result['weights_b']}")
    print(f"Difference in weights (L2 norm): {comp_result['weights_diff']:.4f}")
    print(f"Difference in bias: {comp_result['bias_diff']:.4f}")
    print(f"Difference in training error: {comp_result['train_error_diff']:.4f}")
    print(f"Difference in test error: {comp_result['test_error_diff']:.4f}")
    print("-" * 30)
