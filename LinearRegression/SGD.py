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

def stochastic_gradient_descent(X, y, learning_rate=0.01, tol=1e-6, max_iters=1000):
    m, n = X.shape
    w = np.zeros(n)
    cost_history = []
    
    for t in range(max_iters):
        i = np.random.randint(0, m)
        Xi = X[i, :].reshape(1, -1)
        yi = y[i]
        
        y_pred_i = Xi.dot(w)
        
        gradient = (y_pred_i - yi) * Xi
        
        w_new = w - learning_rate * gradient.flatten()
        
        y_pred = X.dot(w_new)
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        if np.linalg.norm(w_new - w) < tol:
            print(f"SGD Converged after {t} iterations.")
            break
        
        w = w_new
    
    return w, cost_history

learning_rates = [0.01, 0.005, 0.001]
for lr in learning_rates:
    w_sgd, cost_history_sgd = stochastic_gradient_descent(X_train, y_train, learning_rate=lr)
    
    if cost_history_sgd:
        plt.plot(cost_history_sgd, label=f"Learning rate: {lr}")

plt.title("Cost Function over SGD Updates")
plt.xlabel("Number of Updates")
plt.ylabel("Cost")
plt.legend()
plt.show()

def compute_cost(X, y, w):
    m = len(y)
    y_pred = X.dot(w)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost

def normal_equation(X, y):
    w_optimal = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w_optimal

w_optimal = normal_equation(X_train, y_train)

print(f"Optimal weight vector (Analytical): {w_optimal}")

test_cost_optimal = compute_cost(X_test, y_test, w_optimal)
print(f"Cost on test data (Optimal): {test_cost_optimal}")

test_cost_sgd = compute_cost(X_test, y_test, w_sgd)
print(f"Cost on test data (SGD): {test_cost_sgd}")
print(f"Learned weight vector: {w_sgd}")
