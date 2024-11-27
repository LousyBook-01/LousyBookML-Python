"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Standard Scaler Module
This module provides standardization scaling by removing the mean and scaling to unit variance:
- Compute mean and standard deviation from training data
- Transform data by centering and scaling
- Inverse transform to recover original scale

Example:
    >>> from LousyBookML.scalers import StandardScaler
    >>> scaler = StandardScaler()
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)  # Standardized features
"""

import numpy as np
from typing import Optional
from .base import BaseScaler

class StandardScaler(BaseScaler):
    """Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.
    
    Args:
        copy: If True, create a copy of input data during transformations.
              If False, perform in-place scaling.
        with_mean: If True, center the data before scaling.
        with_std: If True, scale the data to unit variance.
        
    Attributes:
        mean_: array of shape (n_features,)
            The mean value for each feature in the training set.
        scale_: array of shape (n_features,)
            The standard deviation for each feature in the training set.
        n_samples_seen_: int
            The number of samples seen by the scaler.
            
    Example:
        >>> scaler = StandardScaler()
        >>> X = np.array([[1, 0], [0, 1], [2, 1]])
        >>> scaler.fit(X)
        >>> print(scaler.mean_)  # [1. 0.667]
        >>> print(scaler.transform(X))
        # [[ 0.    -1.155]
        #  [-1.732  0.577]
        #  [ 1.732  0.577]]
    """
    
    def __init__(self, 
                 copy: bool = True, 
                 with_mean: bool = True,
                 with_std: bool = True):
        super().__init__(copy=copy)
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute the mean and std to be used for later scaling.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self: Returns the scaler instance for method chaining
            
        Raises:
            ValueError: If X is empty or contains NaN values
        """
        X, _ = self._validate_data(X)
        self.n_samples_seen_ = X.shape[0]
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
            
        if self.with_std:
            # Handle single sample case
            if X.shape[0] == 1:
                self.scale_ = np.ones(X.shape[1])
            else:
                # Calculate variance
                X_centered = X - self.mean_ if self.with_mean else X
                # Use population standard deviation (ddof=0)
                self.scale_ = np.std(X_centered, axis=0, ddof=0)
                # Prevent division by zero
                self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
            
        self.is_fitted_ = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features of X according to feature_range.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
            
        Raises:
            ValueError: If the scaler is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")
            
        X, input_is_1d = self._validate_data(X)
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
            
        # Perform scaling
        X_scaled = X
        if self.with_mean:
            X_scaled = X_scaled - self.mean_
        if self.with_std:
            X_scaled = X_scaled / self.scale_
            
        # Preserve input shape for 1D arrays
        if input_is_1d:
            X_scaled = X_scaled.ravel()
            
        return X_scaled if self.copy else X
        
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation.
        
        Args:
            X: Data to inverse transform
            
        Returns:
            Inverse transformed data
            
        Raises:
            ValueError: If the scaler is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")
            
        X, input_is_1d = self._validate_data(X)
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
            
        # Perform inverse scaling
        X_scaled = X
        if self.with_std:
            X_scaled = X_scaled * self.scale_
        if self.with_mean:
            X_scaled = X_scaled + self.mean_
            
        # Preserve input shape for 1D arrays
        if input_is_1d:
            X_scaled = X_scaled.ravel()
            
        return X_scaled if self.copy else X
