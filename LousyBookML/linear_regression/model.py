"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Linear Regression Model Module
This module provides the core linear regression implementation:
- Simple linear regression
- Multiple linear regression
- Polynomial regression
- Regularized regression (L1/L2)
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from .utils import standardize_data, add_polynomial_features
from .losses import mean_squared_error, mean_absolute_error, r2_score

# Re-export all dependencies
__all__ = [
    'LinearRegression',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'standardize_data',
    'add_polynomial_features'
]

class LinearRegression:
    """
    Linear Regression implementation using the normal equation and gradient descent methods.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 method: str = 'normal_equation',
                 standardize: bool = True):
        """
        Initialize Linear Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent
            method: 'normal_equation' or 'gradient_descent'
            standardize: Whether to standardize the input features
            
        Raises:
            ValueError: If method is not 'normal_equation' or 'gradient_descent'
                      If learning_rate is not positive
        """
        if method not in ['normal_equation', 'gradient_descent']:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
            
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.standardize = standardize
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """
        Fit the linear regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples, 1) or (n_samples,)
            
        Returns:
            history: Dictionary containing training metrics
            
        Raises:
            ValueError: If X and y shapes are incompatible
        """
        # Input validation
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Found input samples: {X.shape[0]}, but got target samples: {y.shape[0]}")
            
        # Store original X and y for later use
        X = X.copy()
        y = y.copy()
            
        # Standardize if requested
        if self.standardize:
            X, self.mean, self.std = standardize_data(X)
            
        # Training history
        history = {'loss': []}
            
        if self.method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            history = self._fit_gradient_descent(X, y)
            
        # If standardized, adjust weights and bias back to original scale
        if self.standardize:
            self.weights = self.weights / self.std.reshape(-1, 1)
            self.bias = self.bias - np.sum(self.weights * self.mean)
            self.mean = None
            self.std = None
            
        self.fitted = True
        return history
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using the normal equation method: w = (X^T X)^(-1) X^T y
        """
        n_samples = X.shape[0]
        
        # Add bias term
        X_b = np.c_[np.ones(n_samples), X]
        
        # Use numpy's least squares solver
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        
        # Extract bias and weights
        self.bias = float(theta[0])
        self.weights = theta[1:].reshape(-1, 1)
        
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """
        Fit using gradient descent optimization.
        
        Returns:
            history: Dictionary containing training metrics
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Training history
        history = {'loss': []}
        
        # Use fixed learning rate scaled by number of samples
        lr = self.learning_rate / n_samples
        
        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute loss
            loss = mean_squared_error(y, y_pred)
            history['loss'].append(loss)
            
            # Compute gradients
            error = y_pred - y
            dw = X.T @ error
            db = np.sum(error)
            
            # Update parameters
            self.weights -= lr * dw
            self.bias -= lr * db
            
        return history
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data X.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            predictions: Array of predictions
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Expected {self.weights.shape[0]} features but got {X.shape[1]}")
            
        return X @ self.weights + self.bias
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
