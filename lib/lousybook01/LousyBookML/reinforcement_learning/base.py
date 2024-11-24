"""
LousyBookML - Base DQN Implementation by LousyBook01
www.youtube.com/@LousyBook01

This module implements the base DQN class that other DQN variants build upon.

Made with ❤️ by LousyBook01
"""

import numpy as np
from typing import List, Optional, Tuple
from ..neural_network import NeuralNetwork, LayerConfig

class DQN:
    """Base DQN implementation."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 target_update_freq: int = 100):
        """Initialize DQN.
        
        Args:
            input_dim: Input dimension (state size)
            output_dim: Output dimension (number of actions)
            hidden_dims: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            target_update_freq: How often to update target network
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        # Create networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()
    
    def _build_network(self) -> NeuralNetwork:
        """Build neural network."""
        layers = [LayerConfig(size=self.input_dim)]  # Input layer
        
        # Hidden layers
        prev_size = self.input_dim
        for size in self.hidden_dims:
            layers.append(LayerConfig(size=size, activation='relu'))
            prev_size = size
        
        # Output layer
        layers.append(LayerConfig(size=self.output_dim))
        
        return NeuralNetwork(
            layer_configs=layers,
            learning_rate=self.learning_rate,
            optimizer='rmsprop'
        )
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a given state."""
        return self.q_network.predict(state)
    
    def _update_target_network(self) -> None:
        """Update target network weights with current Q-network weights."""
        self.target_network.weights = [w.copy() for w in self.q_network.weights]
        self.target_network.biases = [b.copy() for b in self.q_network.biases]
    
    def train(self,
             states: np.ndarray,
             actions: np.ndarray,
             rewards: np.ndarray,
             next_states: np.ndarray,
             dones: np.ndarray,
             weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Train the network on a batch of experiences.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of done flags
            weights: Optional importance sampling weights
            
        Returns:
            Tuple of (loss, TD errors)
        """
        if weights is None:
            weights = np.ones_like(rewards)
            
        # Get target Q-values
        next_q_values = self.target_network.predict(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Get current Q-values and compute TD error
        current_q = self.q_network.predict(states)
        batch_indices = np.arange(len(states))
        td_error = target_q - current_q[batch_indices, actions]
        
        # Update Q-network
        target_q_full = current_q.copy()
        target_q_full[batch_indices, actions] = target_q
        loss = self.q_network.train(states, target_q_full, sample_weights=weights)
        
        # Update target network if needed
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss, np.abs(td_error)
