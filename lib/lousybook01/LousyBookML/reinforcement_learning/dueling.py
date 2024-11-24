"""
LousyBookML - Dueling DQN Implementation by LousyBook01
www.youtube.com/@LousyBook01

This module implements the Dueling DQN architecture, which separates the Q-function
into value and advantage streams for better learning of state values.

Made with ❤️ by LousyBook01
"""

import numpy as np
from typing import List, Optional, Tuple
from ..neural_network import NeuralNetwork, LayerConfig
from .base import DQN

class DuelingDQNAgent(DQN):
    """Dueling DQN implementation with separate value and advantage streams."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: List[int] = [64, 64],
                 value_hidden: List[int] = [32],
                 advantage_hidden: List[int] = [32],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 target_update_freq: int = 100):
        """Initialize Dueling DQN.
        
        Args:
            input_dim: Input dimension (state size)
            output_dim: Output dimension (number of actions)
            hidden_layers: List of shared hidden layer sizes
            value_hidden: List of value stream hidden layer sizes
            advantage_hidden: List of advantage stream hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            target_update_freq: How often to update target network
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        # Network architecture parameters
        self.hidden_layers = hidden_layers
        self.value_hidden = value_hidden
        self.advantage_hidden = advantage_hidden
        
        # Create networks
        self.q_network = self._build_dueling_network(learning_rate)
        self.target_network = self._build_dueling_network(learning_rate)
        self._update_target_network()
    
    def _build_dueling_network(self, learning_rate: float) -> NeuralNetwork:
        """Build neural network with dueling architecture."""
        # Shared layers
        layers = [LayerConfig(size=self.input_dim)]  # Input layer
        for size in self.hidden_layers:
            layers.append(LayerConfig(size=size, activation='relu'))
        
        # Value stream
        value_layers = []
        for size in self.value_hidden:
            value_layers.append(LayerConfig(size=size, activation='relu'))
        value_layers.append(LayerConfig(size=1))  # Single value output
        
        # Advantage stream
        advantage_layers = []
        for size in self.advantage_hidden:
            advantage_layers.append(LayerConfig(size=size, activation='relu'))
        advantage_layers.append(LayerConfig(size=self.output_dim))
        
        # Combine streams
        def combine_streams(value: np.ndarray, advantage: np.ndarray) -> np.ndarray:
            """Combine value and advantage streams to get Q-values."""
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            return value + (advantage - np.mean(advantage, axis=1, keepdims=True))
        
        return NeuralNetwork(
            layer_configs=layers + value_layers + advantage_layers,
            learning_rate=learning_rate,
            optimizer='rmsprop',
            stream_combine_fn=combine_streams
        )
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a given state."""
        return self.q_network.predict(state)
    
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
            Tuple of (loss, new priorities)
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
        
        # Return loss and new priorities
        return loss, np.abs(td_error)

class PrioritizedDuelingDQNAgent(DuelingDQNAgent):
    """Dueling DQN with prioritized experience replay."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: List[int] = [64, 64],
                 value_hidden: List[int] = [32],
                 advantage_hidden: List[int] = [32],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 target_update_freq: int = 100,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001):
        """Initialize Prioritized Dueling DQN.
        
        Additional Args:
            alpha: Priority exponent (determines how much prioritization is used)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta during training
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            value_hidden=value_hidden,
            advantage_hidden=advantage_hidden,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_freq=target_update_freq
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
    
    def train(self,
             states: np.ndarray,
             actions: np.ndarray,
             rewards: np.ndarray,
             next_states: np.ndarray,
             dones: np.ndarray,
             weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Train with importance sampling weights.
        
        Additional Args:
            weights: Importance sampling weights for prioritized replay
            
        Returns:
            Tuple of (loss, new priorities)
        """
        loss, td_errors = super().train(states, actions, rewards, next_states, dones, weights)
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Return loss and new priorities (|TD error|^alpha)
        new_priorities = np.abs(td_errors) ** self.alpha
        return loss, new_priorities
