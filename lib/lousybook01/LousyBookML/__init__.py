"""
LousyBookML - A Python machine learning library from scratch made by LousyBook01.

This package provides implementations of:
- Neural Networks with customizable architectures
- Linear Regression with multi-feature support
"""

from .linear_regression import LinearRegression
from .neural_network import NeuralNetwork, LayerConfig

__all__ = ['NeuralNetwork', 'LayerConfig', 'LinearRegression']
