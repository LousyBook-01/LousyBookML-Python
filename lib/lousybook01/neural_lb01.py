import numpy as np
import pickle
import random

class NeuralNetwork:
    """
    A simple neural network implementation with Leaky ReLU activation,
    momentum, and RMSprop optimization.

    Attributes:
        weights1: Weights of the first layer (input to hidden).
        bias1: Biases of the first layer.
        weights2: Weights of the second layer (hidden to output).
        bias2: Biases of the second layer.
        v_w1, v_b1, v_w2, v_b2: Momentum terms for weights and biases.
        s_w1, s_b1, s_w2, s_b2: RMSprop terms for weights and biases.
    """
    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        """
        Initializes the neural network with given sizes.

        Args:
            input_size: Number of features in the input data.
            hidden_size: Number of neurons in the hidden layer.
            output_size: Number of neurons in the output layer.
        """
        # Initialize weights and biases randomly
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

        # Initialize momentum terms for each weight and bias
        self.v_w1 = np.zeros_like(self.weights1)
        self.v_b1 = np.zeros_like(self.bias1)
        self.v_w2 = np.zeros_like(self.weights2)
        self.v_b2 = np.zeros_like(self.bias2)

        # Initialize RMSprop terms for each weight and bias
        self.s_w1 = np.zeros_like(self.weights1)
        self.s_b1 = np.zeros_like(self.bias1)
        self.s_w2 = np.zeros_like(self.weights2)
        self.s_b2 = np.zeros_like(self.bias2)

    def leaky_relu(self, x, alpha=0.01):
        """
        Leaky ReLU activation function.

        Args:
            x: Input value.
            alpha: Slope of the negative part of the function.

        Returns:
            The Leaky ReLU output.
        """
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        """
        Derivative of the Leaky ReLU function.

        Args:
            x: Input value.
            alpha: Slope of the negative part of the function.

        Returns:
            The derivative of the Leaky ReLU output.
        """
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x: Input value.

        Returns:
            The sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.

        Args:
            x: Input value.

        Returns:
            The derivative of the sigmoid output.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_propagation(self, X):
        """
        Performs forward propagation to calculate the network output.

        Args:
            X: Input data.

        Returns:
            The output of the network.
        """
        # Calculate weighted sums and activations for each layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.leaky_relu(self.z1)  # Use Leaky ReLU
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)  # Output layer still uses sigmoid
        return self.a2

    def backward_propagation(self, X, y, learning_rate, momentum=0.9, beta=0.999, epsilon=1e-8):
        """
        Performs backward propagation to update weights and biases.

        Args:
            X: Input data.
            y: Target values.
            learning_rate: Learning rate for weight updates.
            momentum: Momentum factor for weight updates.
            beta: Decay rate for RMSprop.
            epsilon: Small constant added to denominator in RMSprop.
        """
        m = X.shape[0]  # Number of training examples

        # Calculate gradients for output layer
        dcost_da2 = self.a2 - y  # Error term
        da2_dz2 = self.sigmoid(self.z2) * (1 - self.sigmoid(self.z2))  # Derivative of sigmoid
        dz2_dw2 = self.a1.T  # Derivative of weighted sum w.r.t. weights2
        dcost_dw2 = (1 / m) * np.dot(dz2_dw2, dcost_da2 * da2_dz2)
        dcost_db2 = (1 / m) * np.sum(dcost_da2 * da2_dz2, axis=0, keepdims=True)

        # Calculate gradients for hidden layer
        dz2_da1 = self.weights2.T
        dcost_da1 = np.dot(dcost_da2 * da2_dz2, dz2_da1)
        da1_dz1 = self.leaky_relu_derivative(self.z1)  # Derivative of Leaky ReLU
        dz1_dw1 = X.T
        dcost_dw1 = (1 / m) * np.dot(dz1_dw1, dcost_da1 * da1_dz1)
        dcost_db1 = (1 / m) * np.sum(dcost_da1 * da1_dz1, axis=0, keepdims=True)

        # Update weights and biases with momentum and RMSprop
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

    def train(self, X, y, epochs=1000, learning_rate=0.1, debug=False, momentum=0.9, beta=0.999, epsilon=1e-8):
        """
        Trains the neural network on the given data.

        Args:
            X: Input data.
            y: Target values.
            epochs: Number of training iterations.
            learning_rate: Learning rate for weight updates.
            debug: Flag to enable debugging output.
            momentum: Momentum factor for weight updates.
            beta: Decay rate for RMSprop.
            epsilon: Small constant added to denominator in RMSprop.
        """
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)

            # Backward propagation
            self.backward_propagation(X, y, learning_rate, momentum, beta, epsilon)

            # Print cost and debug information every 100 epochs if debug is True
            if debug and epoch % 100 == 0:
                cost = self.cost(y, output)
                print(f"Epoch {epoch}: Cost = {cost}, Weights1 shape: {self.weights1.shape}, Bias1 shape: {self.bias1.shape}, Weights2 shape: {self.weights2.shape}, Bias2 shape: {self.bias2.shape}")

    def cost(self, y, output):
        """
        Calculates the mean squared error cost function.

        Args:
            y: Target values.
            output: Network output.

        Returns:
            The cost value.
        """
        return np.mean(np.square(output - y))

    def predict(self, X, debug=False):
        """
        Makes predictions using the trained network.

        Args:
            X: Input data.
            debug: Flag to enable debugging output.

        Returns:
            Predictions for the given input data.
        """
        predictions = self.forward_propagation(X)
        if debug:
            print(f"Predictions shape: {predictions.shape}")
        return predictions

    def save_model(self, filename):
        """
        Saves the neural network model to a file.

        Args:
            filename: Name of the file to save the model to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Loads a neural network model from a file.

        Args:
            filename: Name of the file to load the model from.

        Returns:
            The loaded neural network model.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

# Example usage:
if __name__ == "__main__":
    # Define input, hidden, and output sizes (with default values)
    # input_size = 2
    # hidden_size = 3
    # output_size = 1
    # 
    # # Create a neural network instance
    # nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn = NeuralNetwork()
    # Define training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the network
    nn.train(X, y, debug=True)

    # # Save the trained model (commented out)
    # nn.save_model('my_trained_model.pkl')

    # Load the saved model 
    # loaded_model = NeuralNetwork.load_model('my_trained_model.pkl')

    # Make predictions using the current neural network (nn)
    for _ in range(5):  # Make 5 predictions
        # Generate random input data
        random_input = [random.randint(0, 1) for _ in range(2)]
        random_input = np.array(random_input).reshape(1, -1)  # Reshape for prediction

        # Make prediction with the random input
        prediction = nn.predict(random_input, debug=False)
        print(f"Input: {random_input}, Prediction: {prediction[0][0]}")