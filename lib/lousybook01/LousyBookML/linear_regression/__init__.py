"""
LousyBookML - Linear Regression implementation by LousyBook01
www.youtube.com/@LousyBook01

A robust implementation of linear regression with support for L2 regularization (Ridge regression).
This module provides efficient matrix-based computations for both training and prediction.

Features:
- L2 regularization support
- Efficient matrix operations using NumPy
- R² score calculation for model evaluation
- Automatic bias term handling

Made with ❤️ by LousyBook01
"""

from .regression import LinearRegression

__all__ = ['LinearRegression']
