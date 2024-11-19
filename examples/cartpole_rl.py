import gym
import numpy as np
import matplotlib.pyplot as plt
from lousybook01.LousyBookML.reinforcement_learning import DQN, PPO

def plot_rewards(rewards, title):
    """Plot training rewards."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

def test_dqn():
    """Test DQN on CartPole."""
    print("\nTesting DQN on CartPole-v1")
    env = gym.make('CartPole-v1')
    
    # Initialize DQN
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dqn = DQN(state_dim=state_dim, action_dim=action_dim, hidden_dims=[64, 64])
    
    # Training loop
    episodes = 500
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Select and perform action
            action = dqn.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store transition and train
            dqn.train(state, action, reward, next_state, terminated or truncated)
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Epsilon: {dqn.epsilon:.2f}")
    
    env.close()
    plot_rewards(rewards, 'DQN Training on CartPole-v1')
    
    # Test the trained agent
    print("\nTesting trained DQN agent...")
    env = gym.make('CartPole-v1', render_mode='human')
    for _ in range(5):  # Run 5 test episodes
        state, _ = env.reset()
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            env.render()
            action = dqn.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
        print(f"Test episode reward: {total_reward}")
    
    env.close()

def test_ppo():
    """Test PPO on CartPole."""
    print("\nTesting PPO on CartPole-v1")
    env = gym.make('CartPole-v1')
    
    # Initialize PPO
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim=state_dim, action_dim=action_dim, hidden_dims=[64, 64])
    
    # Training loop
    episodes = 500
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Select and perform action
            action, action_prob = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store transition
            ppo.store_transition(state, action, reward, next_state, terminated or truncated, action_prob[action])
            
            state = next_state
            total_reward += reward
        
        # Train PPO
        ppo.train()
        rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    env.close()
    plot_rewards(rewards, 'PPO Training on CartPole-v1')
    
    # Test the trained agent
    print("\nTesting trained PPO agent...")
    env = gym.make('CartPole-v1', render_mode='human')
    for _ in range(5):  # Run 5 test episodes
        state, _ = env.reset()
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            env.render()
            action, _ = ppo.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
        print(f"Test episode reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    print("Testing Reinforcement Learning Algorithms on CartPole")
    print("==================================================")
    
    test_dqn()
    test_ppo()
