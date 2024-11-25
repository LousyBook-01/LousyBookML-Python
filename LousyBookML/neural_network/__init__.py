"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Module
This module provides implementations for:
- Feed-forward Neural Networks
- Multiple Activation Functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss Functions (MSE, Binary/Categorical Cross-Entropy)
- Network Training and Utilities

Example:
    >>> from LousyBookML.neural_network import NeuralNetwork
    >>> # Create a neural network for binary classification
    >>> model = NeuralNetwork([
    ...     {'units': 64, 'activation': 'relu'},
    ...     {'units': 32, 'activation': 'relu'},
    ...     {'units': 1, 'activation': 'sigmoid'}
    ... ], loss='binary_crossentropy')
    >>> # Train the model
    >>> model.fit(X_train, y_train, epochs=100, batch_size=32)
    >>> # Make predictions
    >>> predictions = model.predict(X_test)
"""

from .model import NeuralNetwork
from .utils import normalize_data, one_hot_encode, initialize_weights, train_test_split
from .losses import mean_squared_error, binary_cross_entropy, categorical_cross_entropy
from .activations import relu, sigmoid, tanh, softmax

# Make utility functions available as class methods
NeuralNetwork.normalize_data = staticmethod(normalize_data)
NeuralNetwork.one_hot_encode = staticmethod(one_hot_encode)
NeuralNetwork.initialize_weights = staticmethod(initialize_weights)
NeuralNetwork.train_test_split = staticmethod(train_test_split)
