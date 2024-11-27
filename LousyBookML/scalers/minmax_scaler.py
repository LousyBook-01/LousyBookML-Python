"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The MinMax Scaler Module
This module provides feature scaling by normalizing to a fixed range:
- Scale features to a given range (default [0, 1])
- Preserve zero values in sparse data
- Handle custom feature ranges

Example:
    >>> from LousyBookML.scalers import MinMaxScaler
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)  # Features scaled to [0, 1]
"""

import numpy as np
from typing import Optional, Tuple
from .base import BaseScaler

class MinMaxScaler(BaseScaler):
    """Scale features to a fixed range.
    
    Transform features by scaling each feature to a given range.
    The transformation is given by:
        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    where min, max = feature_range.
    
    Args:
        feature_range: Tuple (min, max) giving the range of transformed data.
        copy: If True, create a copy of input data during transformations.
              If False, perform in-place scaling.
        
    Attributes:
        min_: array of shape (n_features,)
            Per feature minimum seen in training data.
        scale_: array of shape (n_features,)
            Per feature scaling factor (X_max - X_min).
        data_min_: array of shape (n_features,)
            Per feature minimum seen in training data.
        data_max_: array of shape (n_features,)
            Per feature maximum seen in training data.
            
    Example:
        >>> scaler = MinMaxScaler(feature_range=(-1, 1))
        >>> X = np.array([[1, 2], [2, 4], [3, 6]])
        >>> scaler.fit(X)
        >>> print(scaler.data_min_)  # [1. 2.]
        >>> print(scaler.data_max_)  # [3. 6.]
        >>> print(scaler.transform(X))
        # [[-1. -1.]
        #  [ 0.  0.]
        #  [ 1.  1.]]
    """
    
    def __init__(self, 
                 feature_range: Tuple[float, float] = (0, 1),
                 copy: bool = True):
        """Initialize MinMaxScaler.
        
        Args:
            feature_range: tuple (min, max), default=(0, 1)
                Desired range of transformed data.
            copy: bool, default=True
                Set to False to perform inplace scaling and avoid a copy.
        """
        super().__init__(copy=copy)
        self.feature_range = feature_range
        if not isinstance(feature_range, tuple) or len(feature_range) != 2:
            raise ValueError("feature_range must be a tuple of length 2")
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Invalid feature range: min >= max")
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None
        
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """Compute the minimum and maximum to be used for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            self: Returns the scaler instance for method chaining
            
        Raises:
            ValueError: If X is empty or contains NaN values
        """
        X, _ = self._validate_data(X)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # If feature has only one value, set range to 1 to avoid division by zero
        self.data_range_[self.data_range_ == 0.0] = 1.0
        
        # Compute scale and min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        
        self.is_fitted_ = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to the given range.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
            
        Raises:
            ValueError: If the scaler is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")
            
        X, input_is_1d = self._validate_data(X)
        
        # Check if any values are outside the training range
        if np.any(X < self.data_min_) or np.any(X > self.data_max_):
            raise ValueError("Input contains values outside of training range.")
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
            
        # Perform scaling
        X_scaled = X
        X_scaled = X_scaled * self.scale_
        X_scaled = X_scaled + self.min_
            
        # Preserve input shape for 1D arrays
        if input_is_1d:
            X_scaled = X_scaled.ravel()
            
        return X_scaled if self.copy else X
        
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the scaling transformation.
        
        Args:
            X: Data to inverse transform
            
        Returns:
            Inverse transformed data
            
        Raises:
            ValueError: If the scaler is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")
            
        X, input_is_1d = self._validate_data(X)
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
            
        # Perform inverse scaling
        X_scaled = X
        X_scaled = (X_scaled - self.min_) / self.scale_
            
        # Preserve input shape for 1D arrays
        if input_is_1d:
            X_scaled = X_scaled.ravel()
            
        return X_scaled if self.copy else X
