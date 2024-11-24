"""
Exploration strategies for reinforcement learning agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""
    
    @abstractmethod
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select an action based on the exploration strategy.
        
        Args:
            q_values: Q-values for each action
            training: Whether we're in training mode
            
        Returns:
            Selected action index
        """
        pass
        
    @abstractmethod
    def update(self) -> None:
        """Update exploration parameters."""
        pass


class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy exploration strategy."""
    
    def __init__(self, epsilon_start: float = 0.99,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """Initialize epsilon-greedy strategy."""
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if not training or np.random.random() > self.epsilon:
            return np.argmax(q_values)
        return np.random.randint(len(q_values))
        
    def update(self) -> None:
        """Update epsilon value."""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end,
                             self.epsilon * self.epsilon_decay)


class UCBExploration(ExplorationStrategy):
    """Upper Confidence Bound exploration strategy."""
    
    def __init__(self, action_dim: int, c: float = 2.0):
        """Initialize UCB exploration.
        
        Args:
            action_dim: Number of possible actions
            c: Exploration coefficient (higher means more exploration)
        """
        self.c = c
        self.action_counts = np.zeros(action_dim)
        self.total_steps = 0
        
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select action using UCB formula."""
        if not training:
            return np.argmax(q_values)
            
        self.total_steps += 1
        
        # Ensure exploration of unvisited actions
        if np.any(self.action_counts == 0):
            return np.random.choice(np.where(self.action_counts == 0)[0])
            
        # UCB formula
        ucb_values = q_values + self.c * np.sqrt(
            np.log(self.total_steps) / self.action_counts
        )
        action = np.argmax(ucb_values)
        self.action_counts[action] += 1
        return action
        
    def update(self) -> None:
        """No update needed as counts are updated in select_action."""
        pass


class BoltzmannExploration(ExplorationStrategy):
    """Boltzmann (Softmax) exploration strategy."""
    
    def __init__(self, temperature_start: float = 1.0,
                 temperature_end: float = 0.01,
                 temperature_decay: float = 0.995):
        """Initialize Boltzmann exploration."""
        self.temperature = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
    
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select action using Boltzmann distribution."""
        if not training:
            return np.argmax(q_values)
            
        # Ensure q_values is 1-dimensional
        q_values = np.ravel(q_values)
        
        # Compute softmax probabilities with numerical stability
        q_shifted = q_values - np.max(q_values)  # For numerical stability
        exp_q = np.exp(q_shifted / max(self.temperature, 1e-8))
        probs = exp_q / np.sum(exp_q)
        
        # Ensure probs is 1-dimensional and valid probability distribution
        probs = np.ravel(probs)
        probs = probs / np.sum(probs)  # Renormalize to ensure sum is 1
        
        return np.random.choice(len(q_values), p=probs)
    
    def update(self) -> None:
        """Update temperature parameter."""
        if self.temperature > self.temperature_end:
            self.temperature = max(self.temperature_end,
                                 self.temperature * self.temperature_decay)


class ThompsonSampling(ExplorationStrategy):
    """Thompson Sampling exploration strategy using Gaussian posteriors."""
    
    def __init__(self, action_dim: int, prior_mean: float = 0.0, prior_std: float = 1.0):
        """Initialize Thompson Sampling.
        
        Args:
            action_dim: Number of possible actions
            prior_mean: Mean of the Gaussian prior
            prior_std: Standard deviation of the Gaussian prior
        """
        self.action_dim = action_dim
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.means = np.full(action_dim, prior_mean)
        self.stds = np.full(action_dim, prior_std)
        self.counts = np.zeros(action_dim)
        
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select action using Thompson Sampling."""
        if not training:
            return np.argmax(q_values)
            
        # Sample from posterior distributions
        samples = np.random.normal(self.means + q_values, self.stds)
        action = np.argmax(samples)
        
        # Update posterior
        self.counts[action] += 1
        self.stds[action] = self.prior_std / np.sqrt(self.counts[action])
        
        return action
        
    def update(self) -> None:
        """No update needed as posteriors are updated in select_action."""
        pass


class NoisyNetworkExploration(ExplorationStrategy):
    """Exploration through parameter space noise."""
    
    def __init__(self, noise_std: float = 0.1, decay: float = 0.995):
        """Initialize Noisy Network exploration.
        
        Args:
            noise_std: Initial standard deviation of the noise
            decay: Decay rate for noise
        """
        self.noise_std = noise_std
        self.initial_std = noise_std
        self.decay = decay
        
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select action by adding noise to Q-values."""
        if not training:
            return np.argmax(q_values)
            
        noisy_q = q_values + np.random.normal(0, self.noise_std, size=q_values.shape)
        return np.argmax(noisy_q)
        
    def update(self) -> None:
        """Decay noise standard deviation."""
        self.noise_std = max(0.01, self.noise_std * self.decay)
