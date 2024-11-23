import pygame
import sys
import numpy as np
from lousybook01.LousyBookML.reinforcement_learning import DQNAgent, EpsilonGreedy
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400  # Reduced height
FPS = 60

# Colors (Dark theme)
BACKGROUND = (18, 18, 18)  # Dark background
TEXT_COLOR = (200, 200, 200)  # Light gray text
WHITE = (255, 255, 255)
CART_COLOR = (70, 130, 180)  # Steel blue
POLE_COLOR = (220, 100, 100)  # Soft red
TRACK_COLOR = (40, 40, 40)  # Slightly lighter than background
TRACK_LIMIT_COLOR = (60, 60, 60)  # Track limit indicators

# CartPole visualization parameters
SCALE = 100  # Pixels per meter
CART_WIDTH = 50  # Smaller cart
CART_HEIGHT = 20
POLE_LENGTH = 80  # Shorter pole
POLE_WIDTH = 6
TRACK_HEIGHT = 3
TRACK_Y = SCREEN_HEIGHT - 60  # Track position

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

class CartPoleVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("CartPole")  # Shorter title
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)  # Smaller font
        
        # Initialize environment and agent
        self.env = SimpleCartPole()
        self.agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dims=[128, 128],
            learning_rate=0.005,
            batch_size=200,
            buffer_size=10000,
            target_update_freq=5,
            exploration_strategy=EpsilonGreedy(epsilon_decay=0.995)
        )
        
        self.episode = 0
        self.total_reward = 0
        self.best_reward = float('-inf')
        self.training = True
        self.state = self.env.reset()
        
        # Add FPS tracking
        self.fps_update_freq = 30  # Update FPS every 30 frames
        self.frame_count = 0
        self.last_time = time.time()
        self.current_fps = 0
        
    def draw_track(self):
        # Draw main track line
        pygame.draw.rect(self.screen, TRACK_COLOR,
                        [50, TRACK_Y, SCREEN_WIDTH - 100, TRACK_HEIGHT])
        
        # Draw track limits
        limit_height = 10
        pygame.draw.rect(self.screen, TRACK_LIMIT_COLOR,
                        [50, TRACK_Y - limit_height//2, 3, limit_height])
        pygame.draw.rect(self.screen, TRACK_LIMIT_COLOR,
                        [SCREEN_WIDTH - 50, TRACK_Y - limit_height//2, 3, limit_height])
        
    def draw_cart_pole(self, state):
        # Convert state to screen coordinates
        cart_x = state[0] * SCALE + SCREEN_WIDTH // 2
        cart_y = TRACK_Y - CART_HEIGHT//2
        
        # Draw cart
        pygame.draw.rect(self.screen, CART_COLOR, 
                        [cart_x - CART_WIDTH//2, cart_y, 
                         CART_WIDTH, CART_HEIGHT])
        
        # Draw pole
        pole_angle = state[2]
        pole_end_x = cart_x + POLE_LENGTH * np.sin(pole_angle)
        pole_end_y = cart_y + CART_HEIGHT//2 - POLE_LENGTH * np.cos(pole_angle)
        pygame.draw.line(self.screen, POLE_COLOR, 
                        (cart_x, cart_y + CART_HEIGHT//2), 
                        (pole_end_x, pole_end_y), 
                        POLE_WIDTH)
        
    def draw_info(self):
        """Draw information overlay."""
        # Draw FPS
        fps_text = self.font.render(f"FPS: {self.current_fps:.1f}", True, WHITE)
        self.screen.blit(fps_text, (10, 10))
        
        # Draw episode and training info
        mode = "Training" if self.training else "Testing"
        episode = self.agent.current_episode
        epsilon = self.agent.exploration_strategy.epsilon
        info_text = self.font.render(
            f"Mode: {mode} | Episode: {episode} | ε: {epsilon:.3f}", True, WHITE
        )
        self.screen.blit(info_text, (10, 40))
        
        # Draw rewards
        reward_text = self.font.render(
            f"Reward: {self.total_reward:.1f} | Best: {self.best_reward:.1f}",
            True, WHITE
        )
        self.screen.blit(reward_text, (10, 70))
            
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.training = not self.training
                        if not self.training:
                            self.agent.exploration_strategy.epsilon = 0
            
            # Get action from agent
            action = self.agent.select_action(self.state)
            next_state, reward, done = self.env.step(action)
            
            # Train agent if in training mode
            loss = 0
            if self.training:
                self.agent.replay_buffer.add(self.state, action, reward, next_state, done)
                loss = self.agent.train(self.state, action, reward, next_state, done)
            
            self.total_reward += reward
            self.best_reward = max(self.best_reward, self.total_reward)
            self.state = next_state
            
            # Reset if episode is done
            if done:
                self.state = self.env.reset()
                self.total_reward = 0
            
            # Draw frame
            self.screen.fill(BACKGROUND)
            self.draw_track()
            self.draw_cart_pole(self.state)
            
            # Update FPS counter
            self.frame_count += 1
            if self.frame_count >= self.fps_update_freq:
                current_time = time.time()
                self.current_fps = self.fps_update_freq / (current_time - self.last_time)
                self.last_time = current_time
                self.frame_count = 0
            
            self.draw_info()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    visualizer = CartPoleVisualizer()
    visualizer.run()