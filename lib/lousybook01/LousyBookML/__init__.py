"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01
"""

__version__ = "0.5.0-beta"

from .neural_network import NeuralNetwork, LayerConfig
from .reinforcement_learning import (
    DQN, DQNAgent, DuelingDQNAgent,
    PrioritizedDQNAgent, PrioritizedDuelingDQNAgent,
    EpsilonGreedy, UCBExploration, BoltzmannExploration,
    ThompsonSampling, NoisyNetworkExploration,
    ReplayBuffer, PrioritizedReplayBuffer
)
from .linear_regression import LinearRegression

__all__ = [
    # Neural Network
    'NeuralNetwork',
    'LayerConfig',
    
    # Reinforcement Learning - Agents
    'DQN',
    'DQNAgent',
    'DuelingDQNAgent',
    'PrioritizedDQNAgent',
    'PrioritizedDuelingDQNAgent',
    
    # Reinforcement Learning - Exploration
    'EpsilonGreedy',
    'UCBExploration',
    'BoltzmannExploration',
    'ThompsonSampling',
    'NoisyNetworkExploration',
    
    # Reinforcement Learning - Memory
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    
    # Linear Regression
    'LinearRegression',
]
