import sys
import numpy as np
import gymnasium as gym
from lousybook01.LousyBookML.reinforcement_learning import DQNAgent, EpsilonGreedy

def main():
    # Initialize environment and agent
    env = gym.make('CartPole-v1')
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        learning_rate=0.005,
        batch_size=64,
        buffer_size=10000,
        target_update_freq=5,
        gamma=0.99,  # Discount factor for future rewards
        exploration_strategy=EpsilonGreedy(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
    )
    
    # Warmup period - collect initial transitions
    print("Warming up replay buffer...")
    state, _ = env.reset()
    for _ in range(1000):  # Collect more random transitions
        action = np.random.randint(2)  # Random action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(state, action, reward, next_state, done)
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    print("\nStarting training...")
    num_episodes = 500  # More episodes
    best_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        done = False
        
        while not done:
            # Get action from agent
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            agent.replay_buffer.add(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.train(state, action, reward, next_state, done)
                if loss is not None:
                    episode_losses.append(loss)
            
            episode_reward += reward
            state = next_state
        
        # Print episode results
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
        print(f"Episode {episode} - Steps: {episode_reward} - Avg Loss: {avg_loss:.4f} - Epsilon: {agent.exploration_strategy.epsilon:.3f}")
        
        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"New best reward: {best_reward}")
    
    env.close()

if __name__ == "__main__":
    main()
