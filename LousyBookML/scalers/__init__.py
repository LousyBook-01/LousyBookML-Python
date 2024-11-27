"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Data Scaling Module
This module provides various data scaling techniques:
- Standard scaling (zero mean and unit variance)
- Min-max scaling (scale to a fixed range)
- Robust scaling (scale using statistics robust to outliers)

Example:
    >>> from LousyBookML.scalers import StandardScaler
    >>> scaler = StandardScaler()
    >>> X_train = np.array([[1, 2], [3, 4]])
    >>> X_scaled = scaler.fit_transform(X_train)
"""

from .base import BaseScaler
from .standard_scaler import StandardScaler
from .minmax_scaler import MinMaxScaler
from .robust_scaler import RobustScaler

__all__ = [
    'BaseScaler',
    'StandardScaler',
    'MinMaxScaler',
    'RobustScaler',
]
