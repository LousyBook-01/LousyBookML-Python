"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Data Scaling Base Module
This module provides the base class for all data scaling operations:
- Base scaler interface
- Common scaling utilities
- Abstract methods for fit and transform operations

Example:
    >>> from LousyBookML.scalers import StandardScaler
    >>> scaler = StandardScaler()
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class BaseScaler(ABC):
    """Base class for all data scaling operations.
    
    All scalers should inherit from this class and implement the
    fit, transform, and inverse_transform methods.
    
    Args:
        copy: If True, create a copy of input data during transformations.
              If False, perform in-place scaling.
        
    Example:
        >>> class CustomScaler(BaseScaler):
        ...     def fit(self, X):
        ...         self.mean_ = np.mean(X, axis=0)
        ...         return self
        ...     def transform(self, X):
        ...         return X - self.mean_
    """
    
    def __init__(self, copy: bool = True):
        self.copy = copy
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseScaler':
        """Learn the scaling parameters from the data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self: Returns the scaler instance for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the scaling transformation.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data of same shape as X
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the scaling transformation.
        
        Args:
            X: Data to inverse transform of shape (n_samples, n_features)
            
        Returns:
            Original scale data of same shape as X
        """
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler and apply transformation in one step.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Transformed data of same shape as X
        """
        return self.fit(X).transform(X)
    
    def _validate_data(self, X: np.ndarray) -> tuple:
        """Validate input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated input data as numpy array and a boolean indicating if input was 1D
            
        Raises:
            ValueError: If input data is empty or not 1D/2D
        """
        # Handle None input
        if X is None:
            raise ValueError("Input array cannot be None")
            
        # Convert to numpy array if not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # Check for empty input
        if X.size == 0:
            raise ValueError("Input array is empty")
            
        # Check for NaN and inf values
        if np.any(~np.isfinite(X)):
            raise ValueError("Input contains NaN or infinity")
        
        # Ensure input is 2D for internal calculations
        input_is_1d = X.ndim == 1
        if input_is_1d:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("Input data must be 1D or 2D array, got shape {}"
                           .format(X.shape))
            
        return X.astype(np.float64), input_is_1d
