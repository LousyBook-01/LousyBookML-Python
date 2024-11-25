"""Loss functions for neural networks."""

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

# For backward compatibility
binary_cross_entropy = binary_crossentropy
categorical_cross_entropy = categorical_crossentropy

# Dictionary mapping loss names to functions
LOSS_FUNCTIONS: Dict[str, callable] = {
    'mean_squared_error': mean_squared_error,
    'mse': mean_squared_error,
    'binary_crossentropy': binary_crossentropy,
    'categorical_crossentropy': categorical_crossentropy
}
