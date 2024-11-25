"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01
"""

__version__ = "0.5.0"

# Linear Regression
from .linear_regression import (
    LinearRegression,
    standardize_data,
    add_polynomial_features,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)