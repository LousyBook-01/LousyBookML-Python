import numpy as np
import pytest
from lousybook01.LousyBookML import DQN, DuelingDQNAgent, PrioritizedReplayBuffer, PrioritizedDQNAgent, PrioritizedDuelingDQNAgent, EpsilonGreedy, DQNAgent

class SimpleCartPole:
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02  # 50Hz simulation
        
        self.steps = 0
        self.max_steps = 500
        self.state = self.reset()
    
    def reset(self):
        # State: [x, x_dot, theta, theta_dot]
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Physics simulation
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Calculate accelerations
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.pole_length
        
        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                   (self.pole_length * (4.0/3.0 - self.pole_mass * cos_theta**2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
        
        # Update state
        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        # Check if episode is done
        done = bool(
            x < -2.4 or x > 2.4 or  # Cart position limits
            theta < -0.21 or theta > 0.21 or  # Pole angle limits (~12 degrees)
            self.steps >= self.max_steps
        )
        
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done

def test_dqn_initialization():
    """Test DQN initialization and basic operations."""
    state_dim = 4
    action_dim = 2
    dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    
    # Test network architectures
    assert len(dqn.q_network.layer_configs) == 4  # input + 2 hidden + output
    assert dqn.q_network.layer_configs[0].size == state_dim
    assert dqn.q_network.layer_configs[-1].size == action_dim
    
    # Test action selection
    state = np.random.rand(state_dim)
    action = dqn.select_action(state, training=False)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < action_dim

def test_dqn_training():
    """Test DQN training on a simple environment."""
    state_dim = 2
    action_dim = 3
    exploration_strategy = EpsilonGreedy(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    dqn = DQNAgent(state_dim=state_dim, action_dim=action_dim, hidden_dims=[16, 16],
                 exploration_strategy=exploration_strategy)
    
    # Generate synthetic training data
    state = np.random.rand(state_dim)
    action = dqn.select_action(state)
    reward = 1.0
    next_state = np.random.rand(state_dim)
    done = False
    
    # Initial training steps
    initial_epsilon = exploration_strategy.epsilon
    for _ in range(10):
        dqn.train(state, action, reward, next_state, done)
    
    # Verify epsilon decay
    assert exploration_strategy.epsilon < initial_epsilon
    
    # Verify target network update
    assert dqn.train_step > 0

def test_cartpole_dqn():
    """Test DQN on a simple CartPole-like environment."""
    class SimpleCartPole:
        def __init__(self):
            self.state = np.random.rand(4)  # position, velocity, angle, angular velocity
            self.steps = 0
            
        def step(self, action):
            # Simplified physics
            self.state += np.random.rand(4) * 0.1
            self.state = np.clip(self.state, -1, 1)
            self.steps += 1
            
            # Simple reward: +1 for staying up, done after 200 steps
            reward = 1.0
            done = self.steps >= 200
            return self.state.copy(), reward, done
            
        def reset(self):
            self.state = np.random.rand(4)
            self.steps = 0
            return self.state.copy()
    
    # Create environment and agent
    env = SimpleCartPole()
    dqn = DQN(state_dim=4, action_dim=2, hidden_dims=[24, 24])
    
    # Training loop
    episodes = 2
    max_steps = 200
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = dqn.select_action(state)
            next_state, reward, done = env.step(action)
            dqn.train(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    # Verify learning
    assert len(total_rewards) == episodes
    assert all(reward > 0 for reward in total_rewards)

def test_dueling_dqn_initialization():
    """Test DuelingDQN initialization and architecture."""
    state_dim = 4
    action_dim = 2
    hidden_dims = [32, 32]
    value_hidden_dims = [16]
    advantage_hidden_dims = [16]
    
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        value_hidden_dims=value_hidden_dims,
        advantage_hidden_dims=advantage_hidden_dims
    )
    
    # Test shared network architecture
    assert len(agent.shared_configs) == len(hidden_dims)
    assert agent.shared_configs[0].input_dim == state_dim
    assert agent.shared_configs[-1].output_dim == hidden_dims[-1]
    
    # Test value stream architecture
    assert len(agent.value_configs) == len(value_hidden_dims) + 1  # +1 for output layer
    assert agent.value_configs[0].input_dim == hidden_dims[-1]
    assert agent.value_configs[-1].output_dim == 1
    
    # Test advantage stream architecture
    assert len(agent.advantage_configs) == len(advantage_hidden_dims) + 1  # +1 for output layer
    assert agent.advantage_configs[0].input_dim == hidden_dims[-1]
    assert agent.advantage_configs[-1].output_dim == action_dim
    
    # Test Q-value computation
    state = np.random.rand(1, state_dim)
    q_values = agent._forward(state, agent.q_network)
    assert q_values.shape == (1, action_dim)

def test_prioritized_replay_buffer():
    """Test PrioritizedReplayBuffer functionality."""
    buffer_size = 100
    buffer = PrioritizedReplayBuffer(buffer_size)
    
    # Test adding transitions
    state_dim = 4
    for i in range(10):
        state = np.random.rand(state_dim)
        next_state = np.random.rand(state_dim)
        buffer.add(state, 0, 1.0, next_state, False)
    
    assert len(buffer) == 10
    assert buffer.priorities[:10].all() == buffer.max_priority
    
    # Test sampling with priorities
    batch_size = 4
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)
    
    assert states.shape == (batch_size, state_dim)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, state_dim)
    assert dones.shape == (batch_size,)
    assert indices.shape == (batch_size,)
    assert weights.shape == (batch_size,)
    
    # Test priority updates
    td_errors = np.array([0.5, 0.1, 0.8, 0.3])
    buffer.update_priorities(indices, td_errors)
    assert buffer.priorities[indices].any() != buffer.max_priority

def test_prioritized_dqn_training():
    """Test PrioritizedDQN training on a simple environment."""
    state_dim = 4
    action_dim = 2
    agent = PrioritizedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[16, 16],
        buffer_size=100,
        batch_size=4
    )
    
    # Generate synthetic training data
    for _ in range(10):
        state = np.random.rand(state_dim)
        action = agent.select_action(state)
        reward = 1.0
        next_state = np.random.rand(state_dim)
        done = False
        agent.replay_buffer.add(state, action, reward, next_state, done)
    
    # Test training step
    loss = agent._perform_train_step()
    assert isinstance(loss, float)
    assert len(agent.replay_buffer) == 10

def test_prioritized_dueling_dqn():
    """Test PrioritizedDuelingDQN on a simple environment."""
    state_dim = 4
    action_dim = 2
    agent = PrioritizedDuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[16, 16],
        value_hidden_dims=[8],
        advantage_hidden_dims=[8],
        buffer_size=100,
        batch_size=4
    )
    
    # Test network architecture
    state = np.random.rand(1, state_dim)
    shared_features = agent.q_network['shared'].predict(state)
    value = agent.q_network['value'].predict(shared_features)
    advantage = agent.q_network['advantage'].predict(shared_features)
    
    assert value.shape == (1, 1)
    assert advantage.shape == (1, action_dim)
    
    # Test training
    for _ in range(10):
        state = np.random.rand(state_dim)
        action = agent.select_action(state)
        reward = 1.0
        next_state = np.random.rand(state_dim)
        done = False
        agent.replay_buffer.add(state, action, reward, next_state, done)
    
    # Test training step
    loss = agent._perform_train_step()
    assert isinstance(loss, float)
    
    # Test action selection
    state = np.random.rand(state_dim)
    action = agent.select_action(state, training=False)
    assert 0 <= action < action_dim

def test_end_to_end_training():
    """End-to-end test of PrioritizedDuelingDQN on a simple environment."""
    class SimpleEnv:
        def __init__(self):
            self.state = np.zeros(4)
            self.steps = 0
        
        def step(self, action):
            self.state = np.random.rand(4)  # Simplified state transition
            self.steps += 1
            reward = 1.0 if action == 1 else 0.0  # Simple reward structure
            done = self.steps >= 10
            return self.state.copy(), reward, done
        
        def reset(self):
            self.state = np.zeros(4)
            self.steps = 0
            return self.state.copy()
    
    # Create environment and agent
    env = SimpleEnv()
    exploration_strategy = EpsilonGreedy(epsilon_start=0.5, epsilon_end=0.01, epsilon_decay=0.995)
    agent = PrioritizedDuelingDQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[16, 16],
        value_hidden_dims=[8],
        advantage_hidden_dims=[8],
        buffer_size=100,
        batch_size=4,
        exploration_strategy=exploration_strategy
    )
    
    # Training loop
    episodes = 2
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent._perform_train_step()
                assert isinstance(loss, float)
            
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    # Verify training completed
    assert len(total_rewards) == episodes
    assert exploration_strategy.epsilon < 0.5  # Epsilon should have decayed

def test_cartpole_realistic():
    """Test DQN on a more realistic CartPole environment."""
    class CartPole:
        def __init__(self):
            # Constants
            self.gravity = 9.8
            self.cart_mass = 1.0
            self.pole_mass = 0.1
            self.pole_length = 0.5
            self.force_mag = 10.0
            self.dt = 0.02
            
            # Thresholds for termination
            self.x_threshold = 2.4
            self.theta_threshold = 12 * np.pi / 180
            
            # State normalization constants
            self.state_means = np.array([0.0, 0.0, 0.0, 0.0])
            self.state_stds = np.array([
                self.x_threshold * 0.5,      # x position
                1.0,                         # x velocity
                self.theta_threshold * 0.5,  # angle
                2.0                          # angular velocity
            ])
            
            # State: [x, x_dot, theta, theta_dot]
            self.state = None
            self.steps = 0
            self.max_steps = 500
            
        def _normalize_state(self, state):
            """Normalize state to have zero mean and unit variance."""
            return (state - self.state_means) / self.state_stds
            
        def step(self, action):
            assert action in [0, 1], "Action should be 0 or 1"
            
            x, x_dot, theta, theta_dot = self.state
            force = self.force_mag if action == 1 else -self.force_mag
            
            # Calculate acceleration using physics equations
            total_mass = self.cart_mass + self.pole_mass
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sin_theta) / total_mass
            theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                       (self.pole_length * (4/3 - self.pole_mass * cos_theta**2 / total_mass))
            x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_theta / total_mass
            
            # Update state using Euler integration
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * x_acc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * theta_acc
            
            self.state = np.array([x, x_dot, theta, theta_dot])
            self.steps += 1
            
            # Check for failure conditions
            done = bool(
                x < -self.x_threshold or
                x > self.x_threshold or
                theta < -self.theta_threshold or
                theta > self.theta_threshold or
                self.steps >= self.max_steps
            )
            
            # Reward: 1 for each step survived
            reward = 1.0 if not done else 0.0
            
            return self._normalize_state(self.state.copy()), reward, done
            
        def reset(self):
            # Initialize state with small random values
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
            self.steps = 0
            return self._normalize_state(self.state.copy())
    
    # Create environment and agent
    env = CartPole()
    exploration_strategy = EpsilonGreedy(
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99  # Faster decay
    )
    
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[128, 128],  # Larger network
        learning_rate=0.0005,    # Lower learning rate
        gamma=0.99,
        exploration_strategy=exploration_strategy,
        target_update_freq=5,    # More frequent target updates
        batch_size=64,           # Larger batch size
        buffer_size=10000,
        double_dqn=True
    )
    
    # Training loop
    episodes = 100              # More episodes
    total_rewards = []
    solved_threshold = 195
    best_reward = 0
    
    print("\nTraining CartPole:")
    print("Episode | Steps | Reward | Epsilon | Best")
    print("-" * 45)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        best_reward = max(best_reward, episode_reward)
        
        # Print episode information
        print(f"{episode:7d} | {env.steps:5d} | {episode_reward:6.1f} | {exploration_strategy.epsilon:7.3f} | {best_reward:4.1f}")
        
        # Early stopping if we solve the environment
        if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= solved_threshold:
            print("\nEnvironment solved!")
            break
    
    # Verify training progress
    assert len(total_rewards) > 0
    assert max(total_rewards) > 50  # Should survive at least 50 steps in best episode
    assert exploration_strategy.epsilon < 1.0  # Epsilon should have decayed

def test_cartpole_visualization():
    # Initialize the environment
    class SimpleCartPole:
        def __init__(self):
            self.state = np.random.rand(4)  # position, velocity, angle, angular velocity
            self.steps = 0
            
        def step(self, action):
            # Simplified physics
            self.state += np.random.rand(4) * 0.1
            self.state = np.clip(self.state, -1, 1)
            self.steps += 1
            
            # Simple reward: +1 for staying up, done after 200 steps
            reward = 1.0
            done = self.steps >= 200
            return self.state.copy(), reward, done
            
        def reset(self):
            self.state = np.random.rand(4)
            self.steps = 0
            return self.state.copy()
    
    env = SimpleCartPole()
    
    # Create the DQN agent with specified parameters
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_layers=[128, 128],
        learning_rate=0.005,
        batch_size=200,
        memory_size=10000,
        target_update_freq=5,
        epsilon_decay=0.995
    )
    
    # Training loop
    num_episodes = 100
    best_reward = float('-inf')
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        best_reward = max(best_reward, total_reward)
        print(f"Episode: {episode}, Steps: {total_reward}, Epsilon: {agent.epsilon:.3f}, Best: {best_reward}")
    
    # Plot the training progress
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('cartpole_training_progress.png')
    plt.close()
    
    # Run a final episode for visualization
    state = env.reset()
    total_reward = 0
    done = False
    states_history = [state]
    
    while not done:
        action = agent.get_action(state, training=False)  # No exploration for visualization
        state, reward, done = env.step(action)
        states_history.append(state)
        total_reward += reward
    
    # Create animation
    import numpy as np
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    
    cart_width = 0.4
    cart_height = 0.2
    pole_length = 1.0
    
    cart = plt.Rectangle((0, 0), cart_width, cart_height, fill=True, color='blue')
    pole, = ax.plot([], [], 'r-', lw=2)
    
    def init():
        ax.add_patch(cart)
        return cart, pole
    
    def animate(i):
        state = states_history[i]
        cart_x = state[0]
        pole_angle = state[2]
        
        # Update cart position
        cart.set_x(cart_x - cart_width/2)
        
        # Update pole position
        pole_x = [cart_x, cart_x + pole_length * np.sin(pole_angle)]
        pole_y = [cart_height, cart_height + pole_length * np.cos(pole_angle)]
        pole.set_data(pole_x, pole_y)
        
        return cart, pole
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(states_history),
                        interval=50, blit=True, repeat=False)
    
    anim.save('cartpole_animation.gif', writer='pillow')
    plt.close()
    
    print(f"Final performance: {total_reward} steps")
    print("Training progress plot saved as 'cartpole_training_progress.png'")
    print("Animation saved as 'cartpole_animation.gif'")
