import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load the training and testing data
train_data = pd.read_csv('bank-note/train.csv')
test_data = pd.read_csv('bank-note/test.csv')

# Extract features and labels
X_train = train_data.iloc[:, :-1].values  # Assuming the last column is the label
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Ensure that labels are in {-1, 1}
y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

# Constants
C_values = [100/873, 500/873, 700/873]  # Adjust according to your dataset size

def dual_objective(alpha, X, y):
    # Compute the dual objective function using matrix operations
    y_alpha = alpha * y
    K = X @ X.T  # Kernel matrix (linear kernel here)
    return 0.5 * y_alpha.T @ K @ y_alpha - np.sum(alpha)

def zero_fun(alpha, y):
    # Equality constraint: sum_i alpha_i * y_i = 0
    return np.dot(alpha, y)

# Function to compute weights and bias from alpha
def compute_w_b(alpha, X, y):
    w = np.sum((alpha * y)[:, None] * X, axis=0)
    sv = (alpha > 1e-5)
    b_candidates = y[sv] - X[sv] @ w
    b = np.mean(b_candidates)
    return w, b

# Main loop for different C values
for C in C_values:
    print(f"Running dual SVM for C = {C}")
    N = X_train.shape[0]
    y = y_train
    X = X_train

    # Initial guess for alpha
    alpha0 = np.zeros(N)

    # Bounds for alpha_i: 0 <= alpha_i <= C
    bounds = [(0, C) for _ in range(N)]

    # Equality constraint
    constraints = {'type': 'eq', 'fun': lambda alpha: zero_fun(alpha, y)}

    # Optimize
    result = minimize(fun=dual_objective,
                      x0=alpha0,
                      args=(X, y),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 1000})

    if not result.success:
        print("Optimization failed:", result.message)
        continue

    # Retrieve optimized alpha
    alpha = result.x

    # Compute weight vector w and bias b
    w, b = compute_w_b(alpha, X, y)

    print(f"Feature weights (w): {w}")
    print(f"Bias (b): {b}")

    # Predictions on the training set
    predictions_train = np.sign(np.dot(X_train, w) + b)
    train_accuracy = np.mean(predictions_train == y_train)
    train_error = 1 - train_accuracy

    # Predictions on the testing set
    predictions_test = np.sign(np.dot(X_test, w) + b)
    test_accuracy = np.mean(predictions_test == y_test)
    test_error = 1 - test_accuracy

    print(f"Training Error: {train_error:.2f}")
    print(f"Testing Error: {test_error:.2f}\n")

    # Compare with primal domain results if available
    # Uncomment and adjust the following lines if you have w_primal and b_primal
    # print(f"Difference in weights: {np.linalg.norm(w - w_primal)}")
    # print(f"Difference in bias: {abs(b - b_primal)}\n")
