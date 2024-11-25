"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Linear Regression Module
This module provides implementations for:
- Simple and Multiple Linear Regression
- Polynomial Regression
- Regularized Regression (L1/L2)
- Regression Metrics and Utilities

Example:
    >>> from LousyBookML.linear_regression import LinearRegression
    >>> # Create and train a simple linear regression model
    >>> model = LinearRegression(learning_rate=0.01)
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 6, 8])
    >>> model.fit(X, y)
    >>> # Make predictions
    >>> predictions = model.predict(np.array([[5], [6]]))
"""

from .model import LinearRegression
from .utils import standardize_data, add_polynomial_features
from .losses import mean_squared_error, r2_score, mean_absolute_error

# Make these available when importing LinearRegression
LinearRegression.standardize_data = staticmethod(standardize_data)
LinearRegression.add_polynomial_features = staticmethod(add_polynomial_features)
LinearRegression.mean_squared_error = staticmethod(mean_squared_error)
LinearRegression.r2_score = staticmethod(r2_score)
LinearRegression.mean_absolute_error = staticmethod(mean_absolute_error)

__all__ = [
    'LinearRegression',
    'standardize_data',
    'add_polynomial_features',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score'
]
