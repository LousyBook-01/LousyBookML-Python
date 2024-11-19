# LousyBookML

A Python machine learning library from scratch made by LousyBook01, implementing neural networks and linear regression.

## Features

- Neural Networks
  - Multiple activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
  - Various optimization algorithms (SGD, Momentum, RMSprop)
  - Batch normalization and dropout
  - Customizable layer configurations

- Linear Regression
  - Multi-feature support
  - L2 regularization
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
print(f"R-squared: {model.r_squared_}")
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
│           ├── linear_regression.py
│           └── neural_network.py
├── tests/
│   ├── test_linear_regression.py
│   └── test_neural_network.py
├── examples/
│   ├── linear_regression_examples.py
│   └── neural_network_examples.py
├── setup.py
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
