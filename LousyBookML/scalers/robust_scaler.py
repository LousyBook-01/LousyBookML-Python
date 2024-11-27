"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Robust Scaler Module
This module provides robust feature scaling using statistics that are robust to outliers:
- Scale features using statistics that are robust to outliers
- Remove median and scale data according to quantile range
- Handle data with outliers effectively

Example:
    >>> from LousyBookML.scalers import RobustScaler
    >>> scaler = RobustScaler(quantile_range=(25.0, 75.0))
    >>> X_train = np.array([[1, 2], [3, 4], [10, 20]])  # Contains outlier
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)  # Scaled without influence of outlier
"""

import numpy as np
from typing import Optional, Tuple
from .base import BaseScaler

class RobustScaler(BaseScaler):
    """Scale features using statistics that are robust to outliers.
    
    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).
    
    Centering and scaling statistics are calculated using:
        - median for centering
        - quantile range for scaling
    
    Args:
        with_centering: If True, center the data before scaling.
        with_scaling: If True, scale the data using the quantile range.
        quantile_range: Tuple (q_min, q_max) for quantile range.
        copy: If True, create a copy of input data during transformations.
              If False, perform in-place scaling.
        
    Attributes:
        center_: array of shape (n_features,)
            The median value for each feature in the training set.
        scale_: array of shape (n_features,)
            The quantile range for each feature in the training set.
            
    Example:
        >>> scaler = RobustScaler()
        >>> X = np.array([[1, 2], [2, 4], [100, 200]])
        >>> scaler.fit(X)
        >>> print(scaler.center_)  # [2. 4.]
        >>> print(scaler.transform(X))
        # [[-0.5 -0.5]
        #  [ 0.   0. ]
        #  [49.  49. ]]  # Outlier less influential
    """
    
    def __init__(self, 
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 copy: bool = True,
                 quantile_interpolation: str = 'midpoint'):
        super().__init__(copy=copy)
        
        # Validate quantile range
        q_min, q_max = quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError(
                "Invalid quantile range. Values must be between 0 and 100 "
                f"with q_min <= q_max. Got ({q_min}, {q_max})"
            )
            
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.quantile_interpolation = quantile_interpolation
        self.center_ = None
        self.scale_ = None
        
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """Compute the quantiles and scale to be used for scaling.
        
        Args:
            X: Training data
            
        Returns:
            self: Returns the scaler instance for method chaining
            
        Raises:
            ValueError: If X is empty or contains NaN values
        """
        X, _ = self._validate_data(X)
        
        q = np.percentile(X,
                         self.quantile_range,
                         axis=0,
                         method=self.quantile_interpolation)
        
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        else:
            self.center_ = np.zeros(X.shape[1])
            
        if self.with_scaling:
            iqr = q[1] - q[0]
            # Set scale to 1 for constant features
            iqr[iqr == 0] = 1.0
            self.scale_ = iqr
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
            raise ValueError("RobustScaler is not fitted. Call fit() first.")
            
        X, input_is_1d = self._validate_data(X)
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
            
        # Perform scaling
        X_scaled = X
        if self.with_centering:
            X_scaled = X_scaled - self.center_
        if self.with_scaling:
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
            raise ValueError("RobustScaler is not fitted. Call fit() first.")
            
        X, input_is_1d = self._validate_data(X)
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
            
        # Perform inverse scaling
        X_scaled = X
        if self.with_scaling:
            X_scaled = X_scaled * self.scale_
        if self.with_centering:
            X_scaled = X_scaled + self.center_
            
        # Preserve input shape for 1D arrays
        if input_is_1d:
            X_scaled = X_scaled.ravel()
            
        return X_scaled if self.copy else X
