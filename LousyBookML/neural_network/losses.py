"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Loss Functions Module
This module provides common loss functions used in neural networks:
- Mean Squared Error (MSE)
- Binary Cross-Entropy
- Categorical Cross-Entropy

Example:
    >>> from LousyBookML.neural_network.losses import mean_squared_error
    >>> y_true = np.array([[1, 0], [0, 1]])
    >>> y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
    >>> loss = mean_squared_error(y_true, y_pred)  # 0.02
"""

import numpy as np
from typing import Dict

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss function.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE loss value
    """
    return np.mean(np.square(y_true - y_pred))

def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """Binary cross-entropy loss function.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid log(0)
        
    Returns:
        Binary cross-entropy loss value
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """Categorical cross-entropy loss function.
    
    Args:
        y_true: True values (one-hot encoded)
        y_pred: Predicted values
        epsilon: Small value to avoid log(0)
        
    Returns:
        Categorical cross-entropy loss value
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def mean_squared_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of mean squared error loss function.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE loss derivative
    """
    return 2 * (y_pred - y_true) / y_true.shape[0]

def binary_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """Derivative of binary cross-entropy loss function.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        Binary cross-entropy loss derivative
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

def categorical_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """Derivative of categorical cross-entropy loss function.
    
    Args:
        y_true: True values (one-hot encoded)
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        Categorical cross-entropy loss derivative
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / (y_pred * y_true.shape[0])

# For backward compatibility
binary_cross_entropy = binary_crossentropy
categorical_cross_entropy = categorical_crossentropy

# Dictionary mapping loss names to (loss_function, derivative) pairs
LOSS_FUNCTIONS: Dict[str, callable] = {
    'mean_squared_error': (mean_squared_error, mean_squared_error_derivative),
    'mse': (mean_squared_error, mean_squared_error_derivative),
    'binary_crossentropy': (binary_crossentropy, binary_crossentropy_derivative),
    'categorical_crossentropy': (categorical_crossentropy, categorical_crossentropy_derivative)
}
