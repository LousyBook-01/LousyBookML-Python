import numpy as np
import pickle
import random

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1, neat=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.neat = neat
        self.initialize_weights()
        self.initialize_momentum_and_rmsprop()

    def initialize_weights(self):
        if self.neat:
            self.weights1 = np.random.uniform(-1, 1, size=(self.input_size, self.hidden_size))
            self.weights2 = np.random.uniform(-1, 1, size=(self.hidden_size, self.output_size))
        else:
            self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)  # He initialization
            self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)  # He initialization

        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def initialize_momentum_and_rmsprop(self):
        self.v_w1 = np.zeros_like(self.weights1)
        self.v_b1 = np.zeros_like(self.bias1)
        self.v_w2 = np.zeros_like(self.weights2)
        self.v_b2 = np.zeros_like(self.bias2)

        self.s_w1 = np.zeros_like(self.weights1)
        self.s_b1 = np.zeros_like(self.bias1)
        self.s_w2 = np.zeros_like(self.weights2)
        self.s_b2 = np.zeros_like(self.bias2)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward_propagation(self, X, y, learning_rate, momentum=0.9, beta=0.999, epsilon=1e-8):
        m = X.shape[0]

        dcost_da2 = self.a2 - y
        da2_dz2 = self.sigmoid_derivative(self.z2)
        dz2_dw2 = self.a1.T
        dcost_dw2 = (1 / m) * np.dot(dz2_dw2, dcost_da2 * da2_dz2)
        dcost_db2 = (1 / m) * np.sum(dcost_da2 * da2_dz2, axis=0, keepdims=True)

        dz2_da1 = self.weights2.T
        dcost_da1 = np.dot(dcost_da2 * da2_dz2, dz2_da1)
        da1_dz1 = self.leaky_relu_derivative(self.z1)
        dz1_dw1 = X.T
        dcost_dw1 = (1 / m) * np.dot(dz1_dw1, dcost_da1 * da1_dz1)
        dcost_db1 = (1 / m) * np.sum(dcost_da1 * da1_dz1, axis=0, keepdims=True)

        self.v_w1 = momentum * self.v_w1 + (1 - momentum) * dcost_dw1
        self.v_b1 = momentum * self.v_b1 + (1 - momentum) * dcost_db1
        self.v_w2 = momentum * self.v_w2 + (1 - momentum) * dcost_dw2
        self.v_b2 = momentum * self.v_b2 + (1 - momentum) * dcost_db2

        self.s_w1 = beta * self.s_w1 + (1 - beta) * np.square(dcost_dw1)
        self.s_b1 = beta * self.s_b1 + (1 - beta) * np.square(dcost_db1)
        self.s_w2 = beta * self.s_w2 + (1 - beta) * np.square(dcost_dw2)
        self.s_b2 = beta * self.s_b2 + (1 - beta) * np.square(dcost_db2)

        self.weights1 -= learning_rate * self.v_w1 / (np.sqrt(self.s_w1) + epsilon)
        self.bias1 -= learning_rate * self.v_b1 / (np.sqrt(self.s_b1) + epsilon)
        self.weights2 -= learning_rate * self.v_w2 / (np.sqrt(self.s_w2) + epsilon)
        self.bias2 -= learning_rate * self.v_b2 / (np.sqrt(self.s_b2) + epsilon)

    def train(self, X, y, epochs=1000, learning_rate=0.1, debug=False, numberOfDebugPrints=10, momentum=0.9, beta=0.999, epsilon=1e-8):
        printPer = int(round(epochs / numberOfDebugPrints))
        print("Printing per ",printPer, " epochs")
        for epoch in range(epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, learning_rate, momentum, beta, epsilon)

            if debug and epoch % printPer == 0 and epoch != 0:
                cost = self.cost(y, output)
                print(f"Epoch {epoch}: Cost = {cost}")
        if debug:
            cost = self.cost(y, output)
            print(f"Epoch {epochs}: Cost = {cost}")
    def cost(self, y, output):
        return np.mean(np.square(output - y))

    def predict(self, X, debug=False):
        predictions = self.forward_propagation(X)
        if debug:
            print(f"Predictions shape: {predictions.shape}")
        return predictions

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

if __name__ == "__main__":
    nn = NeuralNetwork(neat=True)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn.train(X, y, debug=True, learning_rate=0.1, numberOfDebugPrints=5)

    for _ in range(5):
        random_input = [random.randint(0, 1) for _ in range(2)]
        random_input = np.array(random_input).reshape(1, -1)
        prediction = nn.predict(random_input, debug=False)
        print(f"Input: {random_input}, Prediction: {prediction[0][0]}")