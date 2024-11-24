"""
LousyBookML - Neural Network implementation by LousyBook01
www.youtube.com/@LousyBook01

A flexible and efficient neural network implementation supporting various layer configurations
and activation functions. This module provides the building blocks for deep learning models
used throughout LousyBookML.

Features:
- Configurable layer architecture
- Multiple activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
- Various optimization algorithms (SGD, Momentum, RMSprop)
- Batch normalization and dropout
- L1/L2 regularization
- Early stopping

Made with ❤️ by LousyBook01
"""

from .network import NeuralNetwork
from .layers import LayerConfig

__all__ = ['NeuralNetwork', 'LayerConfig']
