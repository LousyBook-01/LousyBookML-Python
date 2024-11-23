import gymnasium as gym
import numpy as np
from sklearn.neural_network import MLPRegressor
import pygame
import random
from collections import deque

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class SklearnDQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64)):
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # Initialize Q-network using scikit-learn MLPRegressor
        self.q_network = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=1,  # We'll do partial_fit instead
            warm_start=True  # Keep weights between fits
        )
        
        # Dummy fit to initialize the model
        dummy_X = np.zeros((1, state_dim))
        dummy_y = np.zeros((1, action_dim))
        self.q_network.fit(dummy_X, dummy_y)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        q_values = self.q_network.predict(state.reshape(1, -1))
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.vstack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Get current Q-values
        current_q = self.q_network.predict(states)
        
        # Get next Q-values
        next_q = self.q_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # Update Q-values for taken actions
        for i in range(self.batch_size):
            current_q[i, actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * max_next_q[i]
        
        # Train on whole batch
        self.q_network.partial_fit(states, current_q)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class CartPoleVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("CartPole - Scikit-learn DQN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 16)
        
        # Create environment
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Create agent
        self.agent = SklearnDQNAgent(self.state_dim, self.action_dim)
        
        self.episode = 0
        self.total_reward = 0
        self.best_reward = -float('inf')
        
        # Scale factors for visualization
        self.cart_scale = 100
        self.pole_scale = 100

    def draw_debug_info(self):
        texts = [
            f"Episode: {self.episode}",
            f"Current Reward: {self.total_reward:.2f}",
            f"Best Reward: {self.best_reward:.2f}",
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Memory Size: {len(self.agent.memory)}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, WHITE)
            self.screen.blit(text_surface, (10, 10 + i * 20))

    def draw_cart_pole(self, state):
        cart_x, cart_y = state[0] * self.cart_scale + SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100
        
        # Draw cart
        pygame.draw.rect(self.screen, RED, 
                        [cart_x - 30, cart_y - 10, 60, 20])
        
        # Draw pole
        pole_angle = state[2]
        pole_length = 100
        pole_end_x = cart_x + np.sin(pole_angle) * pole_length
        pole_end_y = cart_y - np.cos(pole_angle) * pole_length
        pygame.draw.line(self.screen, WHITE, 
                        (cart_x, cart_y), 
                        (pole_end_x, pole_end_y), 6)

    def run_episode(self):
        state, _ = self.env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Clear screen
            self.screen.fill(BLACK)
            
            # Select and perform action
            action = self.agent.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Train agent
            self.agent.train(state, action, reward, next_state, done or truncated)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Draw current state
            self.draw_cart_pole(state)
            self.total_reward = episode_reward
            self.best_reward = max(self.best_reward, episode_reward)
            self.draw_debug_info()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return episode_reward
        
        return episode_reward

    def train(self, num_episodes=1000):
        try:
            for episode in range(num_episodes):
                self.episode = episode + 1
                reward = self.run_episode()
                print(f"Episode {episode + 1}: Reward = {reward:.2f}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RL agent on CartPole")
    parser.add_argument("--episodes", type=int, default=1000,
                      help="Number of episodes to train")
    args = parser.parse_args()
    
    visualizer = CartPoleVisualizer()
    visualizer.train(num_episodes=args.episodes)
