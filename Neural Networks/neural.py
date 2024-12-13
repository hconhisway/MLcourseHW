import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights to zero
        self.weights_1 = np.zeros((input_dim + 1, hidden_dim))
        self.weights_2 = np.zeros((hidden_dim + 1, hidden_dim))
        self.weights_3 = np.zeros((hidden_dim + 1, output_dim))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward_propagation(self, x):
        self.input_with_bias = np.append(x, 1)

        # Layer 1
        self.z1 = np.dot(self.input_with_bias, self.weights_1)
        self.a1 = self.sigmoid(self.z1)

        # Add bias term to Layer 1 output
        self.a1_with_bias = np.append(self.a1, 1)

        # Layer 2
        self.z2 = np.dot(self.a1_with_bias, self.weights_2)
        self.a2 = self.sigmoid(self.z2)

        # Add bias term to Layer 2 output
        self.a2_with_bias = np.append(self.a2, 1)

        # Output Layer
        self.z3 = np.dot(self.a2_with_bias, self.weights_3)
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def backward_propagation(self, x, y):
        # Ensure y is a scalar
        y = np.array([y]) if np.isscalar(y) else y

        # Compute output error
        self.output_error = self.a3 - y  # Derivative of loss with respect to output
        self.output_delta = self.output_error * self.sigmoid_derivative(self.a3)

        # Compute Layer 2 error
        self.z2_error = np.dot(self.output_delta, self.weights_3[:-1].T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.a2)

        # Compute Layer 1 error
        self.z1_error = np.dot(self.z2_delta, self.weights_2[:-1].T)
        self.z1_delta = self.z1_error * self.sigmoid_derivative(self.a1)

        # Gradients
        self.weights_3_gradient = np.outer(self.a2_with_bias, self.output_delta)
        self.weights_2_gradient = np.outer(self.a1_with_bias, self.z2_delta)
        self.weights_1_gradient = np.outer(self.input_with_bias, self.z1_delta)

        return self.weights_1_gradient, self.weights_2_gradient, self.weights_3_gradient

    def update_weights(self, gradients, learning_rate):
        self.weights_1 -= learning_rate * gradients[0]
        self.weights_2 -= learning_rate * gradients[1]
        self.weights_3 -= learning_rate * gradients[2]

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


# Load data
train_X, train_y = load_data("bank-note/train.csv")
test_X, test_y = load_data("bank-note/test.csv")

# Training parameters
gamma_0 = 0.07  # Initial learning rate
d = 3000  # Learning rate decay parameter
epochs = 30
widths = [5, 10, 25, 50, 100]

train_errors = {}
test_errors = {}

for width in widths:
    input_dim = train_X.shape[1]
    hidden_dim = width
    output_dim = 1
    nn = NeuralNetwork(input_dim, hidden_dim, output_dim)

    train_loss = []

    for epoch in range(epochs):
        indices = np.arange(train_X.shape[0])
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]

        for t, (x, y) in enumerate(zip(train_X, train_y)):
            gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)

            nn.forward_propagation(x)

            gradients = nn.backward_propagation(x, y)

            nn.update_weights(gradients, gamma_t)

        loss = 0
        for x, y in zip(train_X, train_y):
            output = nn.forward_propagation(x)
            loss += 0.5 * (output - y) ** 2
        train_loss.append(float(loss / train_X.shape[0]))

    train_errors[width] = float(train_loss[-1])
    test_loss = 0
    for x, y in zip(test_X, test_y):
        output = nn.forward_propagation(x)
        test_loss += 0.5 * (output - y) ** 2
    test_errors[width] = float(test_loss / test_X.shape[0])

    plt.plot(range(epochs), train_loss, label=f"Width {width}")

plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Convergence of Training Loss")
plt.legend()
plt.show()

print("Training and Test Errors:")
for width in widths:
    print(f"Width: {width}, Training Error: {train_errors[width]:.4f}, Test Error: {test_errors[width]:.4f}")
