"""
LousyBookML - A Machine Learning Library from Scratch

This library provides implementations of various machine learning algorithms,
focusing on educational value and code readability. Each implementation includes
detailed documentation and follows best practices for numerical computation.

Modules:
    - linear_regression: Linear regression with L2 regularization
    - neural_network: Flexible neural network with modern features
    - reinforcement_learning: Advanced DQN implementations with modern improvements

Each module is designed to be both educational and practical, with comprehensive
docstrings and type hints to aid understanding.
"""

from .linear_regression import LinearRegression
from .neural_network import NeuralNetwork, LayerConfig
from .reinforcement_learning import (
    DQNAgent as DQN,
    DuelingDQNAgent,
    PrioritizedReplayBuffer,
    PrioritizedDQNAgent,
    PrioritizedDuelingDQNAgent,
    EpsilonGreedy,
    DQNAgent
)

__version__ = '0.1.0'
__author__ = 'LousyBook01'

__all__ = [
    'LinearRegression',
    'NeuralNetwork',
    'LayerConfig',
    'DQN',
    'DuelingDQNAgent',
    'PrioritizedReplayBuffer',
    'PrioritizedDQNAgent',
    'PrioritizedDuelingDQNAgent',
    'EpsilonGreedy',
    'DQNAgent'
]
