"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Activation Functions Module
This module provides common activation functions used in neural networks:
- Rectified Linear Unit (ReLU)
- Sigmoid
- Hyperbolic Tangent (tanh)
- Softmax
- Leaky ReLU
- Linear

Example:
    >>> from LousyBookML.neural_network.activations import relu, sigmoid
    >>> x = np.array([-2, -1, 0, 1, 2])
    >>> relu_output = relu(x)  # array([0, 0, 0, 1, 2])
    >>> sigmoid_output = sigmoid(x)  # array([0.119, 0.269, 0.5, 0.731, 0.881])
"""

import numpy as np
from typing import Union, Callable

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation function.
    
    Computes the element-wise ReLU function:
    f(x) = max(0, x)
    
    Args:
        x: Input array of any shape.
        
    Returns:
        np.ndarray: Array of same shape as input with ReLU activation applied.
        
    Example:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> relu_output = relu(x)
        >>> print(relu_output)  # [0 0 0 1 2]
    """
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function."""
    return (x > 0).astype(int)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.
    
    Computes the element-wise sigmoid function:
    f(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input array of any shape.
        
    Returns:
        np.ndarray: Array of same shape as input with sigmoid activation applied.
        Values are bounded between 0 and 1.
        
    Example:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> sigmoid_output = sigmoid(x)
        >>> print(sigmoid_output)  # [0.119 0.269 0.5 0.731 0.881]
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid activation function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent (tanh) activation function.
    
    Computes the element-wise tanh function:
    f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Args:
        x: Input array of any shape.
        
    Returns:
        np.ndarray: Array of same shape as input with tanh activation applied.
        Values are bounded between -1 and 1.
        
    Example:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> tanh_output = tanh(x)
        >>> print(tanh_output)  # [-0.964 -0.762 0. 0.762 0.964]
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of hyperbolic tangent activation function."""
    return 1 - np.square(np.tanh(x))

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function.
    
    Computes the softmax function for each row of the input array:
    f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    The softmax function normalizes the inputs into a probability distribution,
    where each element is in the range (0, 1) and all elements sum to 1.
    
    Args:
        x: Input array of shape (batch_size, num_features).
        
    Returns:
        np.ndarray: Array of same shape as input with softmax activation applied.
        Each row sums to 1.
        
    Example:
        >>> x = np.array([[1, 2, 3], [4, 5, 6]])
        >>> softmax_output = softmax(x)
        >>> print(softmax_output)  
        # [[0.09 0.244 0.665]
        #  [0.09 0.244 0.665]]
        >>> print(np.sum(softmax_output, axis=1))  # [1. 1.]
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of softmax activation function."""
    s = softmax(x)
    return s * (1 - s)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation function.
    
    Args:
        x (np.ndarray): Input array
        alpha (float): Slope for negative values. Default is 0.01
        
    Returns:
        np.ndarray: Output array after applying Leaky ReLU
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU activation function.
    
    Args:
        x (np.ndarray): Input array
        alpha (float): Slope for negative values. Default is 0.01
        
    Returns:
        np.ndarray: Derivative of Leaky ReLU
    """
    return np.where(x > 0, 1, alpha)

def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function."""
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of linear activation function."""
    return np.ones_like(x)

# Dictionary mapping activation function names to their forward and backward functions
ACTIVATION_FUNCTIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'softmax': (softmax, softmax_derivative),
    'linear': (linear, linear_derivative)
}
