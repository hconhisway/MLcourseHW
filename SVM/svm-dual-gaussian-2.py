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

# Constants
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


# Threshold for identifying support vectors
epsilon = 1e-5

# Store results for comparison
support_vectors_data = {}

# Main loop for training and evaluating SVMs
for gamma in gamma_values:
    print(f"Using gamma = {gamma}")
    K = compute_kernel_matrix(X_train, gamma)  # Compute kernel matrix for the current gamma

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

        support_vectors = np.where(alpha > epsilon)[0]
        support_vectors_data[(gamma, C)] = support_vectors  # Store support vector indices

        # Number of support vectors
        print(f"    Number of Support Vectors: {len(support_vectors)}")

        sv = alpha > epsilon
        b_candidates = y[sv] - np.sum((alpha * y)[:, None] * K[:, sv], axis=0)
        b = np.mean(b_candidates)

        train_predictions = np.sign(np.dot(K, alpha * y) + b)
        train_accuracy = np.mean(train_predictions == y_train)
        train_error = 1 - train_accuracy

        # Compute kernel for test set
        K_test = np.array([[gaussian_kernel(x, x_train, gamma) for x_train in X_train] for x in X_test])
        test_predictions = np.sign(np.dot(K_test, alpha * y) + b)
        test_accuracy = np.mean(test_predictions == y_test)
        test_error = 1 - test_accuracy

        print(f"    Training Error: {train_error:.2f}")
        print(f"    Testing Error: {test_error:.2f}\n")

print("\nOverlapping Support Vectors Analysis (C = 500/873):\n")
selected_C = 500 / 873
gamma_values_sorted = sorted(gamma_values)

for i in range(len(gamma_values_sorted) - 1):
    gamma1 = gamma_values_sorted[i]
    gamma2 = gamma_values_sorted[i + 1]
    sv1 = support_vectors_data[(gamma1, selected_C)]
    sv2 = support_vectors_data[(gamma2, selected_C)]

    overlap_count = len(np.intersect1d(sv1, sv2))  # Compute overlap
    print(f"  Overlap between gamma = {gamma1} and gamma = {gamma2}: {overlap_count} common support vectors")
