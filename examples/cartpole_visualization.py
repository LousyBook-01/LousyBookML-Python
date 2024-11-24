import pygame
import sys
import numpy as np
import gymnasium as gym
from lousybook01.LousyBookML.reinforcement_learning import DQNAgent, EpsilonGreedy, ThompsonSampling, BoltzmannExploration, UCBExploration, NoisyNetworkExploration
import time
import copy

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
FPS = 60

# Colors (Dark theme)
BACKGROUND = (18, 18, 18)
TEXT_COLOR = (200, 200, 200)
WHITE = (255, 255, 255)
AGENT_COLORS = [
    (255, 0, 0),    # Red - Epsilon Greedy
    (0, 255, 0),    # Green - Thompson Sampling
    (0, 0, 255),    # Blue - Boltzmann
    (255, 165, 0),  # Orange - UCB
    (128, 0, 128)   # Purple - Noisy Network
]
TRACK_COLOR = (40, 40, 40)
TRACK_LIMIT_COLOR = (60, 60, 60)
HIGHLIGHT_COLOR = (255, 255, 0)

# CartPole visualization parameters
SCALE = 100
CART_WIDTH = 50
CART_HEIGHT = 20
POLE_LENGTH = 80
POLE_WIDTH = 6
TRACK_HEIGHT = 3
TRACK_Y = SCREEN_HEIGHT - 60

class MultiAgentCartPoleVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-Agent CartPole")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Initialize environments and agents
        self.agents = []
        self.envs = []
        self.states = []
        self.rewards = []
        self.episodes = []
        self.reward_histories = []
        self.episode_losses = []
        
        # Different exploration strategies
        self.agent_names = [
            "Epsilon Greedy",
            "Thompson Sampling",
            "Boltzmann",
            "UCB",
            "Noisy Network"
        ]
        
        for _ in range(5):
            env = gym.make('CartPole-v1')
            state, _ = env.reset()
            self.envs.append(env)
            self.states.append(state)
            self.rewards.append(0)
            self.episodes.append(0)
            self.reward_histories.append([])
            self.episode_losses.append([])
        
        self.agents = [
            DQNAgent(state_dim=4, action_dim=2, hidden_dims=[24], learning_rate=0.001,
                     exploration_strategy=EpsilonGreedy(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)),
            DQNAgent(state_dim=4, action_dim=2, hidden_dims=[24], learning_rate=0.001,
                     exploration_strategy=ThompsonSampling(action_dim=2, prior_std=0.1)),
            DQNAgent(state_dim=4, action_dim=2, hidden_dims=[24], learning_rate=0.001,
                     exploration_strategy=BoltzmannExploration(temperature_start=1.0, temperature_end=0.1)),
            DQNAgent(state_dim=4, action_dim=2, hidden_dims=[24], learning_rate=0.001,
                     exploration_strategy=UCBExploration(action_dim=2, c=0.5)),
            DQNAgent(state_dim=4, action_dim=2, hidden_dims=[24], learning_rate=0.001,
                     exploration_strategy=NoisyNetworkExploration(noise_std=0.1))
        ]
        
        self.training = True
        self.paused = False
        self.focused_agent = None
        self.saved_weights = [None] * len(self.agents)
        self.best_rewards = [float('-inf')] * len(self.agents)
        self.avg_window = 10
        
    def draw_keybinds(self):
        keybinds = [
            "Controls:",
            "SPACE - Pause/Resume",
            "S - Save best weights",
            "L - Load saved weights",
            "1-5 - Focus agent",
            "ESC - Clear focus"
        ]
        
        y = 10
        for bind in keybinds:
            text = self.font.render(bind, True, WHITE)
            self.screen.blit(text, (SCREEN_WIDTH - 200, y))
            y += 25
            
    def render_agent_info(self, agent_idx, x, y, highlight=False):
        agent = self.agents[agent_idx]
        reward = self.rewards[agent_idx]
        avg_reward = (sum(self.reward_histories[agent_idx][-self.avg_window:]) / 
                     len(self.reward_histories[agent_idx][-self.avg_window:])) if self.reward_histories[agent_idx] else 0
        
        color = AGENT_COLORS[agent_idx]
        if highlight:
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, [x-5, y-5, 190, 85], 2)
        
        info = [
            f"Agent {agent_idx+1}: {self.agent_names[agent_idx]}",
            f"Episode: {self.episodes[agent_idx]}",
            f"Reward: {reward:.1f}",
            f"Avg({self.avg_window}): {avg_reward:.1f}",
            f"ε: {agent.exploration_strategy.epsilon:.3f}"
        ]
        
        for i, line in enumerate(info):
            text = self.font.render(line, True, color)
            self.screen.blit(text, (x, y + i*20))
            
    def draw_cart_pole(self, state, agent_idx):
        color = list(AGENT_COLORS[agent_idx])
        if self.focused_agent is not None and agent_idx != self.focused_agent:
            return
        
        if agent_idx == self.reward_histories.index(max(self.reward_histories, key=lambda x: sum(x[-self.avg_window:]) if x else 0)):
            color[3] = 255  # Full opacity for best agent
        
        cart_x = state[0] * SCALE + SCREEN_WIDTH // 2
        cart_y = TRACK_Y - CART_HEIGHT//2
        
        # Draw cart
        cart_surface = pygame.Surface((CART_WIDTH, CART_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(cart_surface, color, [0, 0, CART_WIDTH, CART_HEIGHT])
        self.screen.blit(cart_surface, (cart_x - CART_WIDTH//2, cart_y))
        
        # Draw pole
        pole_surface = pygame.Surface((POLE_LENGTH*2, POLE_LENGTH*2), pygame.SRCALPHA)
        pole_angle = state[2]
        pole_end_x = POLE_LENGTH + POLE_LENGTH * np.sin(pole_angle)
        pole_end_y = POLE_LENGTH - POLE_LENGTH * np.cos(pole_angle)
        pygame.draw.line(pole_surface, color,
                        (POLE_LENGTH, POLE_LENGTH),
                        (pole_end_x, pole_end_y),
                        POLE_WIDTH)
        self.screen.blit(pole_surface, (cart_x - POLE_LENGTH, cart_y - POLE_LENGTH + CART_HEIGHT//2))
        
    def render_agent_info_new(self):
        font = pygame.font.Font(None, 24)
        y_offset = SCREEN_HEIGHT - 150
        
        for i, (agent, color, name) in enumerate(zip(self.agents, AGENT_COLORS, self.agent_names)):
            if self.focused_agent is not None and i != self.focused_agent:
                continue
                
            avg_reward = np.mean(self.reward_histories[i][-10:]) if self.reward_histories[i] else 0
            text = f"{name}: Avg Reward = {avg_reward:.2f}"
            surface = font.render(text, True, color)
            self.screen.blit(surface, (10, y_offset + i * 25))
            
    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_s:
                self.saved_weights = [agent.state_dict() for agent in self.agents]
                print("Saved weights for all agents")
            elif event.key == pygame.K_l:
                for i, weights in enumerate(self.saved_weights):
                    if weights is not None:
                        self.agents[i].load_state_dict(weights)
                print("Loaded saved weights")
            elif event.key == pygame.K_ESCAPE:
                self.focused_agent = None
            elif pygame.K_1 <= event.key <= pygame.K_5:
                agent_idx = event.key - pygame.K_1
                if agent_idx < len(self.agents):
                    self.focused_agent = agent_idx
        
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    for env in self.envs:
                        env.close()
                    sys.exit()
                self.handle_input(event)
            
            if not self.paused:
                for i in range(len(self.agents)):
                    action = self.agents[i].select_action(self.states[i], training=self.training)
                    next_state, reward, terminated, truncated, _ = self.envs[i].step(action)
                    done = terminated or truncated
                    
                    if self.training:
                        loss = self.agents[i].train(self.states[i], action, reward, next_state, done)
                        if loss is not None:
                            self.episode_losses[i].append(loss)
                    
                    self.rewards[i] += reward
                    self.states[i] = next_state
                    
                    if done:
                        print(f"Agent {i+1} - Episode {self.episodes[i]} - Reward: {self.rewards[i]}")
                        self.reward_histories[i].append(self.rewards[i])
                        if self.rewards[i] > self.best_rewards[i]:
                            self.best_rewards[i] = self.rewards[i]
                            print(f"New best reward for Agent {i+1}: {self.best_rewards[i]}")
                        
                        self.states[i], _ = self.envs[i].reset()
                        self.rewards[i] = 0
                        self.episodes[i] += 1
                        self.episode_losses[i] = []
                        
                        if self.training:
                            self.agents[i].end_episode()
            
            # Render
            self.screen.fill(BACKGROUND)
            pygame.draw.rect(self.screen, TRACK_COLOR,
                           [50, TRACK_Y, SCREEN_WIDTH - 100, TRACK_HEIGHT])
            
            # Draw agents
            for i in range(len(self.agents)):
                self.draw_cart_pole(self.states[i], i)
                self.render_agent_info(i, 10, SCREEN_HEIGHT - 120 - i*100,
                                   highlight=i == self.focused_agent)
            
            self.render_agent_info_new()
            self.draw_keybinds()
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    visualizer = MultiAgentCartPoleVisualizer()
    visualizer.run()