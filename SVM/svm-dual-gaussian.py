import numpy as np
import pandas as pd
from scipy.optimize import minimize

train_data = pd.read_csv('bank-note/train.csv')
test_data = pd.read_csv('bank-note/test.csv')

X_train = train_data.iloc[:, :-1].values  # Assuming the last column is the label
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.1, 0.5, 1, 5, 100]  # Hyperparameter for Gaussian kernel


def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)


# Compute the kernel matrix
def compute_kernel_matrix(X, gamma):
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K


# Dual objective function for the SVM
def dual_objective(alpha, K, y):
    y_alpha = alpha * y
    return 0.5 * np.dot(y_alpha, np.dot(K, y_alpha)) - np.sum(alpha)


# Zero constraint for optimization
def zero_fun(alpha, y):
    return np.dot(alpha, y)


results = []
for gamma in gamma_values:
    print(f"Using gamma = {gamma}")
    K = compute_kernel_matrix(X_train, gamma)

    for C in C_values:
        print(f"  Running dual SVM for C = {C}")
        N = X_train.shape[0]
        y = y_train

        # Initial guess for alpha
        alpha0 = np.zeros(N)

        # Bounds for alpha_i: 0 <= alpha_i <= C
        bounds = [(0, C) for _ in range(N)]

        # Equality constraint
        constraints = {'type': 'eq', 'fun': lambda alpha: zero_fun(alpha, y)}

        # Optimize
        result = minimize(fun=dual_objective,
                          x0=alpha0,
                          args=(K, y),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 1000})

        if not result.success:
            print("    Optimization failed:", result.message)
            continue

        alpha = result.x

        sv = alpha > 1e-5
        b_candidates = y[sv] - np.sum((alpha * y)[:, None] * K[:, sv], axis=0)
        b = np.mean(b_candidates)

        # Predictions on the training set
        train_predictions = np.sign(np.sum((alpha * y)[:, None] * K, axis=0) + b)
        train_accuracy = np.mean(train_predictions == y_train)
        train_error = 1 - train_accuracy

        K_test = np.array(
            [[gaussian_kernel(x, x_train, gamma) for x_train in X_train] for x in X_test])  # Shape: (M, N)


        test_predictions = np.sign(np.dot(K_test, alpha * y) + b)  # Fix broadcasting by using dot product

        test_accuracy = np.mean(test_predictions == y_test)
        test_error = 1 - test_accuracy

        print(f"Training Error: {train_error:.2f}")
        print(f"Testing Error: {test_error:.2f}\n")

        results.append((gamma, C, train_error, test_error))
