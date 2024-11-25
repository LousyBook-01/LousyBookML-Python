"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01
"""

__version__ = "0.5.0"

# Import main classes
from .linear_regression.model import LinearRegression
from .neural_network.model import NeuralNetwork

__all__ = ['LinearRegression', 'NeuralNetwork']