import numpy as np
from collections import deque
import random
from .neural_network import NeuralNetwork, LayerConfig

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    """Deep Q-Network implementation."""
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        # Create Q-network
        layers = [LayerConfig(size=state_dim)]
        for dim in hidden_dims:
            layers.append(LayerConfig(size=dim, activation='relu', initialization='he'))
        layers.append(LayerConfig(size=action_dim, activation='linear', initialization='xavier'))
        
        self.q_network = NeuralNetwork(
            layer_configs=layers,
            optimizer='rmsprop',
            learning_rate=0.001,
            batch_norm=True
        )
        
        # Create target network with same architecture
        self.target_network = NeuralNetwork(
            layer_configs=layers,
            optimizer='rmsprop',
            learning_rate=0.001,
            batch_norm=True
        )
        self._update_target_network()
        
        # DQN parameters
        self.buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 8  # Reduced batch size for initial testing
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 10  # Update target network every N steps
        self.train_step = 0
    
    def _update_target_network(self):
        """Copy weights from Q-network to target network."""
        for q_layer, target_layer in zip(self.q_network.weights, self.target_network.weights):
            target_layer[:] = q_layer[:]
        for q_bias, target_bias in zip(self.q_network.biases, self.target_network.biases):
            target_bias[:] = q_bias[:]
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.q_network.layer_configs[-1].size)
        
        state = state.reshape(-1, 1)  # Reshape to (state_dim, 1)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, state, action, reward, next_state, done):
        """Train the DQN on a single transition."""
        # Store transition in replay buffer
        self.buffer.push(state, action, reward, next_state, done)
        
        # Start training only when we have enough samples
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample a batch of transitions
        batch = self.buffer.sample(self.batch_size)
        states = np.vstack([t[0] for t in batch]).T  # Shape: (state_dim, batch_size)
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.vstack([t[3] for t in batch]).T  # Shape: (state_dim, batch_size)
        dones = np.array([t[4] for t in batch], dtype=np.float32)  # Convert to float for multiplication
        
        # Compute target Q-values
        next_q_values = self.target_network.predict(next_states).T  # Shape: (action_dim, batch_size)
        max_next_q_values = np.max(next_q_values, axis=0)  # Shape: (batch_size,)
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Create target matrix for training
        current_q_values = self.q_network.predict(states).T  # Shape: (action_dim, batch_size)
        target_q_matrix = current_q_values.copy()
        
        # Update only the Q-values for the selected actions
        for i in range(self.batch_size):
            target_q_matrix[actions[i], i] = target_q_values[i]
        
        # Train Q-network
        self.q_network.train(states, target_q_matrix, epochs=1, batch_size=self.batch_size)
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class PPO:
    """Proximal Policy Optimization implementation."""
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        # Create actor network (policy)
        actor_layers = [LayerConfig(size=state_dim)]
        for dim in hidden_dims:
            actor_layers.append(LayerConfig(size=dim, activation='relu', initialization='he'))
        actor_layers.append(LayerConfig(size=action_dim, activation='softmax', initialization='xavier'))
        
        self.actor = NeuralNetwork(
            layer_configs=actor_layers,
            optimizer='adam',
            learning_rate=0.0003,
            batch_norm=True
        )
        
        # Create critic network (value function)
        critic_layers = [LayerConfig(size=state_dim)]
        for dim in hidden_dims:
            critic_layers.append(LayerConfig(size=dim, activation='relu', initialization='he'))
        critic_layers.append(LayerConfig(size=1, activation='linear', initialization='xavier'))
        
        self.critic = NeuralNetwork(
            layer_configs=critic_layers,
            optimizer='adam',
            learning_rate=0.001,
            batch_norm=True
        )
        
        # PPO parameters
        self.gamma = 0.99  # Discount factor
        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.batch_size = 32
        self.buffer = []
    
    def select_action(self, state):
        """Select action using current policy."""
        state = state.reshape(-1, 1)  # Reshape to (state_dim, 1)
        action_probs = self.actor.predict(state)  # Shape: (action_dim,)
        action = np.random.choice(len(action_probs[0]), p=action_probs[0])
        return action, action_probs
    
    def store_transition(self, state, action, reward, next_state, done, action_prob):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done, action_prob))
    
    def train(self):
        """Train PPO on collected experiences."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Convert buffer to numpy arrays
        states = np.vstack([t[0] for t in self.buffer]).T  # Shape: (state_dim, batch_size)
        actions = np.array([t[1] for t in self.buffer])
        rewards = np.array([t[2] for t in self.buffer])
        next_states = np.vstack([t[3] for t in self.buffer]).T  # Shape: (state_dim, batch_size)
        dones = np.array([t[4] for t in self.buffer])
        old_action_probs = np.array([t[5] for t in self.buffer])
        
        # Compute advantages
        values = self.critic.predict(states)  # Shape: (batch_size, 1)
        next_values = self.critic.predict(next_states)  # Shape: (batch_size, 1)
        advantages = rewards + self.gamma * next_values[0] * (1 - dones) - values[0]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Train actor
        action_probs = self.actor.predict(states)  # Shape: (batch_size, action_dim)
        ratio = action_probs[np.arange(len(actions)), actions] / old_action_probs
        policy_loss = -np.minimum(
            ratio * advantages,
            np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        )
        
        # Train networks
        self.actor.train(states, action_probs.T, epochs=1, batch_size=self.batch_size)
        self.critic.train(states, values.T, epochs=1, batch_size=self.batch_size)
        
        # Clear buffer
        self.buffer = []
