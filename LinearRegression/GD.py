import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('slump/slump_test.data')

train_data = data.sample(frac=0.75, random_state=42)
test_data = data.drop(train_data.index)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
X_test = (X_test - np.mean(X_test, axis=0)) / (np.std(X_test, axis=0) + 1e-8)
y_train = (y_train - np.mean(y_train)) / (np.std(y_train) + 1e-8)
y_test = (y_test - np.mean(y_test)) / (np.std(y_test) + 1e-8)

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


def gradient_descent(X, y, learning_rate=0.01, tol=1e-6, max_iters=1000):
    m, n = X.shape
    w = np.zeros(n)
    cost_history = []

    for t in range(max_iters):
        y_pred = X.dot(w)
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

        if np.isnan(cost) or np.isinf(cost):
            print(f"Overflow or NaN detected at iteration {t}. Try reducing the learning rate.")
            break

        cost_history.append(cost)

        gradient = (1 / m) * X.T.dot(y_pred - y)

        w_new = w - learning_rate * gradient

        if np.linalg.norm(w_new - w) < tol:
            print(f"Converged after {t} iterations.")
            break

        w = w_new

    return w, cost_history


learning_rates = [0.01, 0.005, 0.001]
for lr in learning_rates:
    w, cost_history = gradient_descent(X_train, y_train, learning_rate=lr)

    print(f"Learning rate: {lr}")
    print(f"Learned weight vector: {w}")

    if cost_history:
        plt.plot(cost_history, label=f"Learning rate: {lr}")

plt.title("Cost Function over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()


def compute_cost(X, y, w):
    m = len(y)
    y_pred = X.dot(w)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


test_cost = compute_cost(X_test, y_test, w)
print(f"Cost on test data: {test_cost}")