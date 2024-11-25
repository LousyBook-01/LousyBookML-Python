"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Linear Regression Utilities Module
This module provides utility functions for linear regression:
- Data standardization
- Feature scaling
- Polynomial feature generation
- Input validation

Example:
    >>> from LousyBookML.linear_regression.utils import standardize_data, add_polynomial_features
    >>> # Standardize features
    >>> X = np.array([[1], [2], [3], [4]])
    >>> X_std, mean, std = standardize_data(X)
    >>> # Generate polynomial features
    >>> X_poly = add_polynomial_features(X, degree=2)  # Adds x² terms
"""

import numpy as np
from typing import Tuple

def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features by removing the mean and scaling to unit variance.
    
    Standardization: z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.
    
    Args:
        X: Input features of shape (n_samples, n_features).
        
    Returns:
        Tuple containing:
            - Standardized features
            - Feature means
            - Feature standard deviations
        
    Example:
        >>> X = np.array([[1], [2], [3], [4]])
        >>> X_std, mean, std = standardize_data(X)
        >>> print(f"Mean: {mean[0]:.2f}, Std: {std[0]:.2f}")  # Mean: 2.50, Std: 1.12
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def add_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Generate polynomial features up to the specified degree.
    
    For each feature x, generates powers up to x^degree.
    For example, if degree=2: [x] -> [x, x²]
    
    Args:
        X: Input features of shape (n_samples, n_features).
        degree: Maximum polynomial degree.
        
    Returns:
        np.ndarray: Array with original and polynomial features.
        
    Example:
        >>> X = np.array([[1], [2], [3]])
        >>> X_poly = add_polynomial_features(X, degree=2)
        >>> print(X_poly)  # array([[1, 1], [2, 4], [3, 9]])
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))
    
    for d in range(1, degree + 1):
        for feat in range(n_features):
            X_poly = np.column_stack((X_poly, X[:, feat] ** d))
            
    return X_poly[:, 1:]  # Remove the ones column
