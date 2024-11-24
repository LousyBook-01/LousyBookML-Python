"""
Deep Q-Network (DQN) agent implementations.
"""

import numpy as np
from typing import List, Dict, Optional
from ..neural_network import NeuralNetwork, LayerConfig
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .exploration import ExplorationStrategy, EpsilonGreedy

class DQNAgent:
    """Deep Q-Network agent implementation.
    
    Features:
    - Experience replay for stable learning
    - Target network to reduce instability
    - Double DQN option to mitigate overestimation bias
    - Flexible exploration strategies
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 exploration_strategy: ExplorationStrategy = EpsilonGreedy(),
                 target_update_freq: int = 100,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 double_dqn: bool = True,
                 min_buffer_size: int = 1000):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            exploration_strategy: Exploration strategy to use
            target_update_freq: Frequency of target network updates
            batch_size: Size of training batches
            buffer_size: Size of replay buffer
            double_dqn: Whether to use double DQN
            min_buffer_size: Minimum number of transitions before training starts
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.min_buffer_size = min_buffer_size
        self.episode_rewards = []
        self.episode_steps = 0
        self.current_episode = 0
        self.total_steps = 0
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Create Q-networks
        layer_configs = []
        
        # Input layer
        layer_configs.append(LayerConfig(
            input_dim=state_dim,
            output_dim=state_dim,  # Input layer same size as state
            activation='linear',
            initialization='he'
        ))
        
        # Hidden layers
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layer_configs.append(LayerConfig(
                input_dim=prev_dim,
                output_dim=hidden_dim,
                activation='relu',
                initialization='he'
            ))
            prev_dim = hidden_dim
        
        # Output layer
        layer_configs.append(LayerConfig(
            input_dim=prev_dim,
            output_dim=action_dim,
            activation='linear',
            initialization='he'
        ))
        
        # Create main and target networks
        self.q_network = NeuralNetwork(
            layer_configs=layer_configs,
            optimizer='adam',
            learning_rate=learning_rate
        )
        self.target_network = NeuralNetwork(
            layer_configs=layer_configs,
            optimizer='adam',
            learning_rate=learning_rate
        )
        
        # Initialize target network with same weights
        self._update_target_network()
        
    def train_step(self) -> float:
        """Get the current training step."""
        return self._perform_train_step() if len(self.replay_buffer) >= self.min_buffer_size else 0.0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using exploration strategy.
        
        Args:
            state: Current state
            training: Whether to use exploration
            
        Returns:
            Selected action index
        """
        state = np.array(state).reshape(1, -1)
        q_values = self.q_network.forward(state)
        return self.exploration_strategy.select_action(q_values[0], training)
        
    def _perform_train_step(self) -> float:
        """Perform one training step using a batch from replay buffer.
        
        Returns:
            Loss value from training step
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Get current Q-values
        current_q = self.q_network.forward(states)
        
        # Get next Q-values from target network
        next_q = self.target_network.forward(next_states)
        
        if self.double_dqn:
            # Get actions from main network
            next_actions = np.argmax(self.q_network.forward(next_states), axis=1)
            # Get Q-values for those actions from target network
            next_q_values = next_q[np.arange(len(next_actions)), next_actions]
        else:
            next_q_values = np.max(next_q, axis=1)
        
        # Compute target Q-values
        target_q = current_q.copy()
        target_q[np.arange(len(actions)), actions] = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Update Q-network
        loss = self.q_network.backward(states, target_q)
        
        # Periodically update target network
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
            
        return loss
        
    def train(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> float:
        """Train the DQN agent on a single transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Loss value from training
        """
        # Add transition to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Update counters
        self.total_steps += 1
        self.episode_steps += 1
        
        # Perform training step if buffer has enough samples
        return self.train_step()
        
    def end_episode(self) -> None:
        """Called at the end of each episode to update exploration."""
        self.current_episode += 1
        self.episode_rewards.append(self.episode_steps)
        self.episode_steps = 0
        self.exploration_strategy.update()
        
    def _update_target_network(self) -> None:
        """Update target network weights with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def save(self, filepath: str) -> None:
        """Save the DQN model weights.
        
        Args:
            filepath: Path to save the model
        """
        self.q_network.save(filepath)
        
    def load(self, filepath: str) -> None:
        """Load the DQN model weights.
        
        Args:
            filepath: Path to the saved model
        """
        self.q_network.load(filepath)
        self._update_target_network()


class PrioritizedDQNAgent(DQNAgent):
    """DQN agent with prioritized experience replay.
    
    This implementation combines the standard DQN with prioritized experience replay
    for more efficient learning by focusing on important transitions.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 exploration_strategy: ExplorationStrategy = EpsilonGreedy(),
                 target_update_freq: int = 100,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 double_dqn: bool = True,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 min_buffer_size: int = 1000):
        """Initialize prioritized DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            exploration_strategy: Exploration strategy to use
            target_update_freq: Frequency of target network updates
            batch_size: Size of training batches
            buffer_size: Size of replay buffer
            double_dqn: Whether to use double DQN
            alpha: How much prioritization to use
            beta: Initial importance sampling weight
            beta_increment: How much to increase beta over time
            min_buffer_size: Minimum number of transitions before training starts
        """
        # Initialize everything except the replay buffer
        super().__init__(state_dim, action_dim, hidden_dims, learning_rate,
                        gamma, exploration_strategy, target_update_freq,
                        batch_size, buffer_size, double_dqn, min_buffer_size)
        
        # Replace standard replay buffer with prioritized version
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size, alpha, beta, beta_increment
        )
        
    def _perform_train_step(self) -> float:
        """Perform one training step using prioritized experience replay.
        
        Returns:
            Loss value from training step
        """
        # Sample batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Get current Q-values
        current_q = self.q_network.forward(states)
        
        # Get next Q-values from target network
        next_q = self.target_network.forward(next_states)
        
        if self.double_dqn:
            # Get actions from main network
            next_actions = np.argmax(self.q_network.forward(next_states), axis=1)
            # Get Q-values for those actions from target network
            next_q_values = next_q[np.arange(len(next_actions)), next_actions]
        else:
            next_q_values = np.max(next_q, axis=1)
        
        # Compute target Q-values
        target_q = current_q.copy()
        target_q[np.arange(len(actions)), actions] = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute TD errors for updating priorities
        td_errors = np.abs(target_q[np.arange(len(actions)), actions] - current_q[np.arange(len(actions)), actions])
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Weight the loss by importance sampling weights
        loss = self.q_network.backward(states, target_q, sample_weights=weights)
        
        # Periodically update target network
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
            
        return loss
