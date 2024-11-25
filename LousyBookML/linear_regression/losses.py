"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Linear Regression Loss Functions Module
This module provides loss functions for regression models:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) Score
- Adjusted R-squared Score

Example:
    >>> from LousyBookML.linear_regression.losses import mean_squared_error, r2_score
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mse = mean_squared_error(y_true, y_pred)
    >>> r2 = r2_score(y_true, y_pred)
"""

import numpy as np
from typing import Union, Optional

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean squared error loss between true and predicted values.
    
    MSE = (1/n) * Σ(y_true - y_pred)²
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        float: Mean squared error loss value.
        
    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> mse = mean_squared_error(y_true, y_pred)
        >>> print(f"MSE Loss: {mse:.4f}")  # MSE Loss: 0.4375
    """
    return np.mean(np.square(y_true - y_pred))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the root mean squared error loss between true and predicted values.
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        float: Root mean squared error loss value.
        
    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> rmse = root_mean_squared_error(y_true, y_pred)
        >>> print(f"RMSE Loss: {rmse:.4f}")  # RMSE Loss: 0.6614
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute error loss between true and predicted values.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        float: Mean absolute error loss value.
        
    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> mae = mean_absolute_error(y_true, y_pred)
        >>> print(f"MAE Loss: {mae:.4f}")  # MAE Loss: 0.5000
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared (coefficient of determination) score.
    
    R² = 1 - (MSE(y_true, y_pred) / Var(y_true))
    where MSE is mean squared error and Var is variance.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        float: R-squared score in range (-∞, 1]. Higher values indicate better fit.
        
    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> r2 = r2_score(y_true, y_pred)
        >>> print(f"R² Score: {r2:.4f}")  # R² Score: 0.9486
    """
    mse = mean_squared_error(y_true, y_pred)
    var = np.var(y_true)
    return 1 - (mse / var if var != 0 else 0)

def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Calculate the adjusted R-squared score.
    
    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    where n is number of samples and p is number of features.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        n_features: Number of features used in the model.
        
    Returns:
        float: Adjusted R-squared score. Penalizes the addition of non-informative features.
        
    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> adj_r2 = adjusted_r2_score(y_true, y_pred, n_features=2)
        >>> print(f"Adjusted R² Score: {adj_r2:.4f}")  # Adjusted R² Score: 0.8972
    """
    n_samples = len(y_true)
    if n_samples <= n_features + 1:
        return float('nan')  # Not enough samples for the number of features
    
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
