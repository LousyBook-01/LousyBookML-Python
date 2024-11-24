"""
Deep Q-Network (DQN) Implementation

This module implements a DQN agent with the following features:
- Experience replay for stable learning
- Target network to reduce instability
- Double DQN option to mitigate overestimation bias
- Epsilon-greedy exploration
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from collections import deque
import random
from .neural_network import NeuralNetwork, LayerConfig

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
        """Return current size of the buffer."""
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
        """Add a new transition to the buffer with maximum priority."""
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
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Get sampling probabilities from priorities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)
        
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
        # Update max priority only for new priorities
        self.max_priority = priorities.max()

class ExplorationStrategy:
    """Base class for exploration strategies."""
    def select_action(self, q_values: np.ndarray, training: bool = True) -> int:
        """Select an action based on the exploration strategy.
        
        Args:
            q_values: Q-values for each action
            training: Whether we're in training mode
            
        Returns:
            Selected action index
        """
        raise NotImplementedError
    
    def update(self):
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
        if not training:
            return np.argmax(q_values)
        
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        return np.argmax(q_values)
    
    def update(self):
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
        
        # If some actions haven't been tried, try them first
        if np.any(self.action_counts == 0):
            return np.random.choice(np.where(self.action_counts == 0)[0])
        
        # Compute UCB values
        ucb_values = q_values + self.c * np.sqrt(
            np.log(self.total_steps) / self.action_counts
        )
        
        # Select action with highest UCB value
        action = np.argmax(ucb_values)
        
        # Update counts
        self.action_counts[action] += 1
        self.total_steps += 1
        
        return action
    
    def update(self):
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

class DQNAgent:
    """Deep Q-Network agent implementation.
    
    Features:
    - Experience replay for stable learning
    - Target network to reduce instability
    - Double DQN option to mitigate overestimation bias
    - Epsilon-greedy exploration
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
    
    @property
    def train_step(self) -> int:
        """Get the current training step."""
        return self.total_steps
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using exploration strategy.
        
        Args:
            state: Current state
            training: Whether to use exploration
            
        Returns:
            Selected action index
        """
        q_values = self.q_network.predict(state.reshape(1, -1))
        return self.exploration_strategy.select_action(q_values, training)
    
    def _perform_train_step(self) -> float:
        """Perform one training step using a batch from replay buffer.
        
        Returns:
            Loss value from training step
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Get current Q values
        current_q_values = self.q_network.predict(states)
        
        # Get next Q values from target network
        next_q_values = self.target_network.predict(next_states)
        
        if self.double_dqn:
            # Get actions from main network
            next_actions = np.argmax(self.q_network.predict(next_states), axis=1)
            # Get Q values for those actions from target network
            max_next_q_values = next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Standard DQN: use max Q value from target network
            max_next_q_values = np.max(next_q_values, axis=1)
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Update only the Q values for the actions taken
        target = current_q_values.copy()
        target[np.arange(self.batch_size), actions] = target_q_values
        
        # Train the network with only 1 epoch
        history = self.q_network.train(states, target, max_iter=1)
        loss = history['loss'][-1] if history['loss'] else 0.0
        
        # Update target network if needed
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss
    
    def train(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> Optional[float]:
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
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        
        # Perform training step
        loss = self._perform_train_step()
        
        # Update target network if needed
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
        
        # Update total steps
        self.total_steps += 1
        
        return loss
    
    def end_episode(self):
        """Called at the end of each episode to update exploration."""
        self.exploration_strategy.update()
        self.current_episode += 1
    
    def _update_target_network(self) -> None:
        """Update target network weights with current Q-network weights."""
        for target_layer, q_layer in zip(self.target_network.weights, self.q_network.weights):
            target_layer[:] = q_layer
        for target_bias, q_bias in zip(self.target_network.biases, self.q_network.biases):
            target_bias[:] = q_bias
    
    def save(self, filepath: str) -> None:
        """Save the DQN model weights.
        
        Args:
            filepath: Path to save the model
        """
        self.q_network.save_model(filepath)
    
    def load(self, filepath: str) -> None:
        """Load the DQN model weights.
        
        Args:
            filepath: Path to the saved model
        """
        self.q_network.load_model(filepath)
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
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch with priorities and importance weights
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-values
        if self.double_dqn:
            # Double DQN: use main network to select actions, target network to evaluate
            next_q_values_main = self.q_network.predict(next_states)
            next_actions = np.argmax(next_q_values_main, axis=1)
            next_q_values = self.target_network.predict(next_states)
            next_q_values = next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Regular DQN: use target network for both selection and evaluation
            next_q_values = self.target_network.predict(next_states)
            next_q_values = np.max(next_q_values, axis=1)
        
        # Compute target values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q-values and create targets
        current_q_values = self.q_network.predict(states)
        targets = current_q_values.copy()
        targets[np.arange(self.batch_size), actions] = target_q_values
        
        # Compute TD errors for priority update
        td_errors = target_q_values - current_q_values[np.arange(self.batch_size), actions]
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Train the network with importance sampling weights
        history = self.q_network.train(states, targets, sample_weight=weights, max_iter=1)
        loss = history['loss'][-1] if history['loss'] else 0.0
        
        # Update exploration strategy
        self.exploration_strategy.update()
        
        # Update steps
        self.total_steps += 1
        
        # Update target network if needed
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss

class DuelingDQNAgent(DQNAgent):
    """Dueling DQN implementation that separates state value and advantage streams.
    
    The dueling architecture helps learn which states are valuable without having to
    learn the effect of each action at each state, leading to faster training and
    better policy evaluation.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 value_hidden_dims: List[int] = [32],
                 advantage_hidden_dims: List[int] = [32],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 exploration_strategy: ExplorationStrategy = EpsilonGreedy(),
                 target_update_freq: int = 100,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 double_dqn: bool = True,
                 min_buffer_size: int = 1000):
        """Initialize Dueling DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of shared hidden layer dimensions
            value_hidden_dims: List of value stream hidden layer dimensions
            advantage_hidden_dims: List of advantage stream hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            exploration_strategy: Exploration strategy to use
            target_update_freq: Frequency of target network updates
            batch_size: Size of training batches
            buffer_size: Size of replay buffer
            double_dqn: Whether to use double DQN
            min_buffer_size: Minimum number of transitions before training starts
        """
        # Don't call parent's __init__ since we're replacing the network architecture
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
        
        # Create shared layers
        layer_configs = []
        prev_dim = state_dim
        
        # Create shared layers based on hidden_dims
        for hidden_dim in hidden_dims:
            layer_configs.append(LayerConfig(
                input_dim=prev_dim,
                output_dim=hidden_dim,
                activation='relu',
                initialization='he'
            ))
            prev_dim = hidden_dim
        
        # Value stream
        value_configs = []
        value_dim = prev_dim
        for hidden_dim in value_hidden_dims:
            value_configs.append(LayerConfig(
                input_dim=value_dim,
                output_dim=hidden_dim,
                activation='relu',
                initialization='he'
            ))
            value_dim = hidden_dim
        
        # Final value layer (scalar)
        value_configs.append(LayerConfig(
            input_dim=value_dim,
            output_dim=1,
            activation='linear',
            initialization='he'
        ))
        
        # Advantage stream
        advantage_configs = []
        advantage_dim = prev_dim
        for hidden_dim in advantage_hidden_dims:
            advantage_configs.append(LayerConfig(
                input_dim=advantage_dim,
                output_dim=hidden_dim,
                activation='relu',
                initialization='he'
            ))
            advantage_dim = hidden_dim
        
        # Final advantage layer (one per action)
        advantage_configs.append(LayerConfig(
            input_dim=advantage_dim,
            output_dim=action_dim,
            activation='linear',
            initialization='he'
        ))
        
        # Create combined network configurations
        self.shared_configs = layer_configs
        self.value_configs = value_configs
        self.advantage_configs = advantage_configs
        
        # Create main and target networks
        self.q_network = self._create_dueling_network(learning_rate)
        self.target_network = self._create_dueling_network(learning_rate)
        
        # Initialize target network with same weights
        self._update_target_network()
    
    @property
    def train_step(self) -> int:
        """Get the current training step."""
        return self.total_steps
    
    def _increment_steps(self):
        """Increment the training step counter."""
        self.total_steps += 1
        
    def _create_dueling_network(self, learning_rate: float) -> Dict[str, NeuralNetwork]:
        """Create the three components of the dueling network architecture.
        
        Args:
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Dictionary containing the three network components
        """
        return {
            'shared': NeuralNetwork(
                layer_configs=self.shared_configs,
                optimizer='adam',
                learning_rate=learning_rate
            ),
            'value': NeuralNetwork(
                layer_configs=self.value_configs,
                optimizer='adam',
                learning_rate=learning_rate
            ),
            'advantage': NeuralNetwork(
                layer_configs=self.advantage_configs,
                optimizer='adam',
                learning_rate=learning_rate
            )
        }
    
    def _forward(self, states: np.ndarray, network: Dict[str, NeuralNetwork]) -> np.ndarray:
        """Forward pass through the dueling network architecture.
        
        Args:
            states: Batch of states
            network: Dictionary containing the three network components
            
        Returns:
            Q-values for each action
        """
        # Forward through shared layers
        shared_features = network['shared'].predict(states)
        
        # Get state values and advantages
        values = network['value'].predict(shared_features)
        advantages = network['advantage'].predict(shared_features)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantages_mean = np.mean(advantages, axis=1, keepdims=True)
        q_values = values + (advantages - advantages_mean)
        
        return q_values
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using exploration strategy with dueling architecture.
        
        Args:
            state: Current state
            training: Whether to use exploration
            
        Returns:
            Selected action index
        """
        q_values = self._forward(state.reshape(1, -1), self.q_network)
        return self.exploration_strategy.select_action(q_values, training)
    
    def _perform_train_step(self) -> float:
        """Perform one training step using a batch from replay buffer.
        
        Returns:
            Average loss value across all network components
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-values using dueling architecture
        if self.double_dqn:
            # Double DQN: use main network to select actions, target network to evaluate
            next_q_values_main = self._forward(next_states, self.q_network)
            next_actions = np.argmax(next_q_values_main, axis=1)
            next_q_values = self._forward(next_states, self.target_network)
            next_q_values = next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Regular DQN: use target network for both selection and evaluation
            next_q_values = self._forward(next_states, self.target_network)
            next_q_values = np.max(next_q_values, axis=1)
        
        # Compute target values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q-values and create targets
        current_q_values = self._forward(states, self.q_network)
        targets = current_q_values.copy()
        targets[np.arange(self.batch_size), actions] = target_q_values
        
        # Train each network component
        shared_features = self.q_network['shared'].predict(states)
        
        # Train value stream
        value_targets = np.mean(targets, axis=1, keepdims=True)
        value_history = self.q_network['value'].train(
            shared_features, value_targets, max_iter=1, batch_size=self.batch_size
        )
        
        # Train advantage stream
        advantage_targets = targets - value_targets
        advantage_history = self.q_network['advantage'].train(
            shared_features, advantage_targets, max_iter=1, batch_size=self.batch_size
        )
        
        # Update exploration strategy
        self.exploration_strategy.update()
        
        # Update steps
        self._increment_steps()
        
        # Update target network if needed
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
        
        # Return average loss across value and advantage streams
        value_loss = value_history.get('loss', [0.0])[-1]
        advantage_loss = advantage_history.get('loss', [0.0])[-1]
        return (value_loss + advantage_loss) / 2
    
    def train(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> Optional[float]:
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
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        
        # Perform training step
        loss = self._perform_train_step()

        if done:
            epsilon = self.exploration_strategy.epsilon
            print(f"Episode {self.current_episode} - Steps: {self.episode_steps} - Total Steps: {self.total_steps} - Loss: {loss:.4f} - Epsilon: {epsilon:.3f}")
            self.current_episode += 1
            self.episode_steps = 0
            
        return loss
    
    def end_episode(self):
        """Called at the end of each episode to update exploration."""
        self.exploration_strategy.update()
        self.current_episode += 1
    
    def _update_target_network(self) -> None:
        """Update target network weights with current Q-network weights."""
        # Update shared layers
        for target_layer, q_layer in zip(self.target_network['shared'].weights,
                                       self.q_network['shared'].weights):
            target_layer[:] = q_layer
        for target_bias, q_bias in zip(self.target_network['shared'].biases,
                                     self.q_network['shared'].biases):
            target_bias[:] = q_bias
        
        # Update value stream
        for target_layer, q_layer in zip(self.target_network['value'].weights,
                                       self.q_network['value'].weights):
            target_layer[:] = q_layer
        for target_bias, q_bias in zip(self.target_network['value'].biases,
                                     self.q_network['value'].biases):
            target_bias[:] = q_bias
        
        # Update advantage stream
        for target_layer, q_layer in zip(self.target_network['advantage'].weights,
                                       self.q_network['advantage'].weights):
            target_layer[:] = q_layer
        for target_bias, q_bias in zip(self.target_network['advantage'].biases,
                                     self.q_network['advantage'].biases):
            target_bias[:] = q_bias
    
    def save(self, filepath: str) -> None:
        """Save the dueling DQN model weights.
        
        Args:
            filepath: Base path to save the model components
        """
        # Save each network component with a suffix
        self.q_network['shared'].save_model(f"{filepath}_shared")
        self.q_network['value'].save_model(f"{filepath}_value")
        self.q_network['advantage'].save_model(f"{filepath}_advantage")
    
    def load(self, filepath: str) -> None:
        """Load the dueling DQN model weights.
        
        Args:
            filepath: Base path to the saved model components
        """
        # Load each network component
        self.q_network['shared'].load_model(f"{filepath}_shared")
        self.q_network['value'].load_model(f"{filepath}_value")
        self.q_network['advantage'].load_model(f"{filepath}_advantage")
        self._update_target_network()

class PrioritizedDuelingDQNAgent(DuelingDQNAgent):
    """Advanced DQN implementation combining dueling architecture with prioritized replay.
    
    This agent combines the benefits of both dueling networks (better value estimation)
    and prioritized experience replay (more efficient learning) for state-of-the-art
    performance in deep reinforcement learning tasks.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 value_hidden_dims: List[int] = [32],
                 advantage_hidden_dims: List[int] = [32],
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
        """Initialize prioritized dueling DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of shared hidden layer dimensions
            value_hidden_dims: List of value stream hidden layer dimensions
            advantage_hidden_dims: List of advantage stream hidden layer dimensions
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
        # Initialize the dueling network architecture
        super().__init__(state_dim, action_dim, hidden_dims, value_hidden_dims,
                        advantage_hidden_dims, learning_rate, gamma, exploration_strategy,
                        target_update_freq, batch_size, buffer_size, double_dqn, min_buffer_size)
        
        # Replace standard replay buffer with prioritized version
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size, alpha, beta, beta_increment
        )
    
    def _perform_train_step(self) -> float:
        """Perform one training step using prioritized experience replay.
        
        Returns:
            Average loss value across all network components
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch with priorities and importance weights
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-values using dueling architecture
        if self.double_dqn:
            # Double DQN: use main network to select actions, target network to evaluate
            next_q_values_main = self._forward(next_states, self.q_network)
            next_actions = np.argmax(next_q_values_main, axis=1)
            next_q_values = self._forward(next_states, self.target_network)
            next_q_values = next_q_values[np.arange(self.batch_size), next_actions]
        else:
            # Regular DQN: use target network for both selection and evaluation
            next_q_values = self._forward(next_states, self.target_network)
            next_q_values = np.max(next_q_values, axis=1)
        
        # Compute target values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Get current Q-values and create targets
        current_q_values = self._forward(states, self.q_network)
        targets = current_q_values.copy()
        targets[np.arange(self.batch_size), actions] = target_q_values
        
        # Compute TD errors for priority update
        td_errors = target_q_values - current_q_values[np.arange(self.batch_size), actions]
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Train each network component with importance sampling weights
        shared_features = self.q_network['shared'].predict(states)
        
        # Train value stream
        value_targets = np.mean(targets, axis=1, keepdims=True)
        value_history = self.q_network['value'].train(
            shared_features, value_targets,
            max_iter=1, batch_size=self.batch_size,
            sample_weight=weights
        )
        
        # Train advantage stream
        advantage_targets = targets - value_targets
        advantage_history = self.q_network['advantage'].train(
            shared_features, advantage_targets,
            max_iter=1, batch_size=self.batch_size,
            sample_weight=weights
        )
        
        # Update exploration strategy
        self.exploration_strategy.update()
        
        # Update steps
        self._increment_steps()
        
        # Update target network if needed
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
        
        # Return average loss across value and advantage streams
        value_loss = value_history.get('loss', [0.0])[-1]
        advantage_loss = advantage_history.get('loss', [0.0])[-1]
        return (value_loss + advantage_loss) / 2

class LayerConfig:
    """Configuration for a neural network layer."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = 'linear',
                 initialization: str = 'he',
                 dropout_rate: float = 0.0,
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.0,
                 batch_norm: bool = False):
        """Initialize layer configuration.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function ('linear', 'relu', 'sigmoid', etc.)
            initialization: Weight initialization method ('he', 'xavier', etc.)
            dropout_rate: Dropout rate (0.0 means no dropout)
            l1_reg: L1 regularization coefficient
            l2_reg: L2 regularization coefficient
            batch_norm: Whether to use batch normalization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initialization = initialization
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_norm = batch_norm
        
        # Add size attribute for compatibility
        self.size = output_dim