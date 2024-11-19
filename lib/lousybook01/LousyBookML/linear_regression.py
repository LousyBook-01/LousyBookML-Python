"""
Linear Regression Implementation from scratch by LousyBook01.

Features:
    - Multi-feature support
    - L2 regularization
    - Statistical metrics (R-squared, MSE)
    - Robust numerical computations
    - Input validation
    - Flexible model configuration
"""

import numpy as np
from typing import Optional, Tuple, Union

class LinearRegression:
    """Linear Regression with optional regularization and statistical metrics.
    
    Attributes:
        coef_: array-like, shape (n_features,)
            Estimated coefficients for the linear regression problem.
        intercept_: float
            Independent term in the linear model (y-intercept).
        r_squared_: float
            R-squared score of the model.
        mse_: float
            Mean squared error of the model.
    """
    
    def __init__(self, alpha: float = 0.0, fit_intercept: bool = True):
        """Initialize LinearRegression.
        
        Args:
            alpha: Regularization strength (L2 regularization). Default is 0.0 (no regularization).
            fit_intercept: Whether to calculate and include intercept. Default is True.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.r_squared_ = None
        self.mse_ = None
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and format input data.
        
        Args:
            X: Input features of shape (n_samples, n_features) or (n_samples,)
            y: Target values of shape (n_samples,)
            
        Returns:
            Tuple of formatted X and y arrays
        """
        # Convert to numpy arrays if not already
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Reshape X if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check dimensions
        if X.ndim != 2:
            raise ValueError("X must be 2D array or 1D array, got shape {0}".format(X.shape))
        if y.ndim != 1:
            raise ValueError("y must be 1D array, got shape {0}".format(y.shape))
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
            
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """Fit linear regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features) or (n_samples,)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Returns an instance of self.
        """
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape
        
        # Add intercept column if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1
        
        # Solve normal equations with regularization
        # (X^T X + αI)β = X^T y
        if self.alpha > 0:
            # Add regularization term (exclude intercept from regularization if fitting it)
            reg_matrix = self.alpha * np.eye(n_features)
            if self.fit_intercept:
                reg_matrix[0, 0] = 0  # Don't regularize intercept
        else:
            reg_matrix = 0
            
        try:
            beta = np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse for ill-conditioned matrices
            beta = np.linalg.pinv(X.T @ X + reg_matrix) @ X.T @ y
        
        # Extract intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = beta
        
        # Calculate fit metrics
        y_pred = self.predict(X[:, 1:] if self.fit_intercept else X)
        self.mse_ = np.mean((y - y_pred) ** 2)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        self.r_squared_ = 1 - (ss_res / ss_tot)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear regression model.
        
        Args:
            X: Input data of shape (n_samples, n_features) or (n_samples,)
            
        Returns:
            array-like: Predicted values
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X.shape[1]} features, but model was trained with {len(self.coef_)} features")
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R-squared score of the model.
        
        Args:
            X: Test samples
            y: True values for X
            
        Returns:
            score: R-squared score
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'r_squared_': self.r_squared_,
            'mse_': self.mse_,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept
        }
