"""
LousyBookML - Reinforcement Learning implementation by LousyBook01
www.youtube.com/@LousyBook01

This module provides a comprehensive implementation of various reinforcement learning algorithms,
including DQN (Deep Q-Network), Prioritized DQN, and multiple exploration strategies.

Key Components:
- DQN and Prioritized DQN agents for deep reinforcement learning
- Various exploration strategies (Epsilon-Greedy, UCB, Boltzmann, Thompson Sampling)
- Replay buffer implementations for experience replay
- Prioritized replay buffer for improved learning from important experiences

Made with ❤️ by LousyBook01
"""

from .agents import DQNAgent, PrioritizedDQNAgent
from .base import DQN
from .dueling import DuelingDQNAgent, PrioritizedDuelingDQNAgent
from .exploration import (
    EpsilonGreedy, UCBExploration, BoltzmannExploration,
    ThompsonSampling, NoisyNetworkExploration
)
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

__all__ = [
    'DQNAgent',
    'PrioritizedDQNAgent',
    'EpsilonGreedy',
    'UCBExploration',
    'BoltzmannExploration',
    'ThompsonSampling',
    'NoisyNetworkExploration',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'DQN',
    'DuelingDQNAgent',
    'PrioritizedDuelingDQNAgent',
]
