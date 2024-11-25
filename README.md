# LousyBookML

A comprehensive Python machine learning library from scratch made by LousyBook01, implementing neural networks, reinforcement learning, and linear regression.

Made with ❤️ by [LousyBook01](www.youtube.com/@LousyBook01) a 13 year old kid, who loves to code.
Some help was given by Codeium's new Windsurf IDE with Cascade and Claude.


## Features

- Reinforcement Learning
  - Deep Q-Network (DQN) implementation
  - Prioritized Experience Replay
  - Multiple exploration strategies:
    - Epsilon-Greedy
    - UCB (Upper Confidence Bound)
    - Boltzmann Exploration
    - Thompson Sampling
    - Noisy Network Exploration
  - Flexible replay buffer system
  - Compatible with Gymnasium environments

- Neural Networks
  - Multiple activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
  - Various optimization algorithms (SGD, Momentum, RMSprop)
  - Batch normalization and dropout
  - Customizable layer configurations

- Linear Regression
  - Multi-feature support
  - L2 regularization (Ridge Regression)
  - Statistical metrics (R-squared, MSE)
  - Robust numerical computations

## Installation

```bash
# Install basic package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Linear Regression Example

```python
import numpy as np
from lousybook01.LousyBookML import LinearRegression

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7]]))
print(f"Predictions: {predictions}")
print(f"R-squared: {model.score(X, y)}")
```

### Neural Network Example

```python
from lousybook01.LousyBookML import NeuralNetwork, LayerConfig

# Create layer configurations
layers = [
    LayerConfig(size=2),  # Input layer
    LayerConfig(size=4, activation='relu'),  # Hidden layer
    LayerConfig(size=1, activation='sigmoid')  # Output layer
]

# Create and train model
model = NeuralNetwork(
    layer_configs=layers,
    optimizer='momentum',
    learning_rate=0.1
)

# Train on XOR problem
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

model.train(X, y, epochs=1000, batch_size=4)
```

### Reinforcement Learning Example

```python
import gymnasium as gym
from lousybook01.LousyBookML import DQNAgent, EpsilonGreedy

# Create environment
env = gym.make('CartPole-v1')

# Create DQN agent with epsilon-greedy exploration
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    exploration_strategy=EpsilonGreedy(
        initial_epsilon=1.0,
        min_epsilon=0.01,
        decay_rate=0.995
    )
)

# Train the agent
episodes = 1000
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Project Structure

```
LousyBookML-Python/
├── lib/
│   └── lousybook01/
│       └── LousyBookML/
│           ├── __init__.py
│           ├── reinforcement_learning/
│           │   ├── __init__.py
│           │   ├── agents.py
│           │   ├── exploration.py
│           │   └── replay_buffer.py
│           ├── neural_network/
│           │   ├── __init__.py
│           │   └── layers.py
│           └── linear_regression/
│               ├── __init__.py
│               └── regression.py
├── tests/
│   ├── test_reinforcement_learning.py
│   ├── test_neural_network.py
│   └── test_linear_regression.py
├── examples/
│   ├── reinforcement_learning_examples.py
│   ├── neural_network_examples.py
│   └── linear_regression_examples.py
└── README.md
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

Made with ❤️ by LousyBook01 (www.youtube.com/@LousyBook01)
