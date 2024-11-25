"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Utilities Module
This module provides utility functions for neural networks:
- Data Normalization
- One-Hot Encoding
- Weight Initialization
- Data Splitting

Example:
    >>> from LousyBookML.neural_network.utils import normalize_data, one_hot_encode
    >>> # Normalize features
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_norm, mean, std = normalize_data(X)
    >>> # One-hot encode labels
    >>> y = np.array([0, 1, 2, 1])
    >>> y_onehot = one_hot_encode(y)  # Creates 3x4 matrix
"""

import numpy as np
from typing import Tuple, Union

def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize input data using z-score normalization.
    
    Standardization formula: z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.
    
    Args:
        X: Input data array of shape (n_samples, n_features).
        
    Returns:
        Tuple containing:
        - np.ndarray: Normalized data
        - np.ndarray: Mean values used for normalization
        - np.ndarray: Standard deviation values used for normalization
        
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X_norm, mean, std = normalize_data(X)
        >>> print(f"Mean: {mean}")  # Mean: [3. 4.]
        >>> print(f"Std: {std}")   # Std: [1.633 1.633]
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std, mean, std

def to_categorical(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    """Convert class vector (integers from 0 to num_classes) to binary class matrix.
    
    Args:
        y: Class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: Total number of classes. If None, this will be inferred
            from the max value in y.
            
    Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def one_hot_encode(y: np.ndarray, num_classes: Union[int, None] = None) -> np.ndarray:
    """Convert class labels to one-hot encoded format.
    
    Creates a binary matrix where each row corresponds to a label
    and contains a 1 in the column corresponding to that class.
    
    Args:
        y: Array of class labels (integers).
        num_classes: Number of unique classes. If None, determined from data.
        
    Returns:
        np.ndarray: One-hot encoded array of shape (n_samples, num_classes).
        
    Example:
        >>> y = np.array([0, 1, 2, 1])
        >>> one_hot = one_hot_encode(y)
        >>> print(one_hot)
        # [[1 0 0]
        #  [0 1 0]
        #  [0 0 1]
        #  [0 1 0]]
    """
    if num_classes is None:
        num_classes = len(np.unique(y))
    return np.eye(num_classes)[y.astype(int)]

def initialize_weights(input_dim: int, output_dim: int, method: str = 'he') -> np.ndarray:
    """Initialize weight matrix using various initialization methods.
    
    Supported methods:
    - He initialization: W ~ N(0, sqrt(2/n_in))
    - Xavier/Glorot: W ~ N(0, sqrt(1/n_in))
    - Random: W ~ N(0, 0.01)
    
    Args:
        input_dim: Input dimension (number of features).
        output_dim: Output dimension (number of neurons).
        method: Initialization method ('he', 'xavier', or 'random').
        
    Returns:
        np.ndarray: Initialized weight matrix of shape (input_dim, output_dim).
        
    Example:
        >>> W = initialize_weights(input_dim=3, output_dim=2, method='he')
        >>> print(W.shape)  # (3, 2)
        >>> print(np.std(W))  # ~0.816 (≈ sqrt(2/3))
    """
    if method == 'he':
        return np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
    elif method == 'xavier':
        return np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
    else:  # Random initialization
        return np.random.randn(input_dim, output_dim) * 0.01

def train_test_split(X: np.ndarray, 
                    y: np.ndarray, 
                    test_size: float = 0.2, 
                    shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets.
    
    Args:
        X: Input features of shape (n_samples, n_features).
        y: Target values of shape (n_samples,) or (n_samples, n_targets).
        test_size: Proportion of data to use for testing (between 0 and 1).
        shuffle: Whether to shuffle the data before splitting.
        
    Returns:
        Tuple containing:
        - np.ndarray: Training features
        - np.ndarray: Testing features
        - np.ndarray: Training targets
        - np.ndarray: Testing targets
        
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        >>> y = np.array([0, 1, 0, 1, 0])
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        >>> print(f"Training samples: {len(X_train)}")  # Training samples: 3
        >>> print(f"Testing samples: {len(X_test)}")   # Testing samples: 2
    """
    assert 0 < test_size < 1, "test_size must be between 0 and 1"
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        
    test_samples = int(n_samples * test_size)
    train_indices = indices[test_samples:]
    test_indices = indices[:test_samples]
    
    return (X[train_indices], X[test_indices],
            y[train_indices], y[test_indices])
