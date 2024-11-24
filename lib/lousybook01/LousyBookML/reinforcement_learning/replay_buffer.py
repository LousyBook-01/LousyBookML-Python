"""
Replay buffer implementations for reinforcement learning.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple

class ReplayBuffer:
    """Experience replay buffer for DQN training.
    
    Stores and samples transitions (state, action, reward, next_state, done) for training.
    Can be extended to support prioritized experience replay.
    """
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add a new transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer that samples transitions based on TD error.
    
    This implementation uses proportional prioritization with importance sampling
    weights to correct for the bias introduced by non-uniform sampling.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta: Importance sampling weight (0 = no correction, 1 = full correction)
            beta_increment: How much to increase beta over time
            epsilon: Small constant to add to priorities to ensure non-zero sampling
        """
        super().__init__(capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Add a new transition with maximum priority."""
        idx = len(self.buffer) % self.capacity
        super().add(state, action, reward, next_state, done)
        self.priorities[idx] = self.max_priority
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions based on their priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) == 0:
            return None
            
        # Get sampling probabilities from priorities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            td_errors: TD errors for the transitions
        """
        priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))
