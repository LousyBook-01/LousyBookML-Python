"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Optimizers Module
This module provides optimization algorithms for neural networks:
- Stochastic Gradient Descent (SGD) with Momentum
- Adam Optimizer
- RMSprop Optimizer

Example:
    >>> from LousyBookML.neural_network.optimizers import SGD, Adam
    >>> # Create SGD optimizer with momentum
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
    >>> # Create Adam optimizer
    >>> optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .activations import ACTIVATION_FUNCTIONS

class SGD:
    """Stochastic Gradient Descent optimizer with momentum.
    
    Implements the momentum update rule:
    v = momentum * v - learning_rate * gradient
    w = w + v
    
    Args:
        learning_rate: Learning rate (step size) for optimization.
        momentum: Momentum coefficient (0 <= momentum < 1).
        nesterov: Whether to use Nesterov momentum.
        
    Example:
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> optimizer.initialize(model.get_parameters())
        >>> for epoch in range(epochs):
        ...     predictions = model.forward(X)
        ...     optimizer.step(model.layers, X, y, predictions)
    """
    
    def __init__(self, learning_rate: float = 0.01, 
                 momentum: float = 0.0,
                 nesterov: bool = False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}
    
    def initialize(self, params: Dict[str, np.ndarray]) -> None:
        """Initialize velocity for each parameter.
        
        Args:
            params: Dictionary of parameters to optimize.
        """
        self.velocities = {key: np.zeros_like(value) for key, value in params.items()}
    
    def compute_gradients(self, layers: List[Any], X: np.ndarray, 
                         y: np.ndarray, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients using backpropagation.
        
        Args:
            layers: List of network layers.
            X: Input data.
            y: Target values.
            predictions: Network predictions.
            
        Returns:
            Dictionary containing gradients for each parameter.
        """
        m = X.shape[0]
        n_layers = len(layers)
        grads = {}
        
        # Compute output layer error
        error = predictions - y
        
        # Backpropagate through layers
        for i in reversed(range(n_layers)):
            layer = layers[i]
            
            # Compute activation gradient
            if i == n_layers - 1:
                delta = error * layer.activation_derivative(layer.output)
            else:
                delta = np.dot(delta, layers[i+1].weights.T) * layer.activation_derivative(layer.output)
            
            # Compute weight and bias gradients
            grads[f'dW{i+1}'] = np.dot(layer.input.T, delta) / m
            grads[f'db{i+1}'] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Compute batch normalization gradients if applicable
            if layer.batch_norm:
                grads[f'dgamma{i+1}'] = np.sum(delta * layer.normalized_input, axis=0, keepdims=True) / m
                grads[f'dbeta{i+1}'] = np.sum(delta, axis=0, keepdims=True) / m
        
        return grads
    
    def step(self, layers: List[Any], X: np.ndarray, 
             y: np.ndarray, predictions: np.ndarray) -> None:
        """Update parameters using SGD with momentum.
        
        Args:
            layers: List of network layers.
            X: Input data.
            y: Target values.
            predictions: Network predictions.
        """
        # Get parameter dictionary
        params = {}
        for i, layer in enumerate(layers):
            params[f'W{i+1}'] = layer.weights
            params[f'b{i+1}'] = layer.bias
            if layer.batch_norm:
                params[f'gamma{i+1}'] = layer.gamma
                params[f'beta{i+1}'] = layer.beta
        
        # Initialize velocities if needed
        if not self.velocities:
            self.initialize(params)
        
        # Compute gradients
        grads = self.compute_gradients(layers, X, y, predictions)
        
        # Update parameters with momentum
        for key in params:
            if self.nesterov:
                # Nesterov momentum update
                v_prev = self.velocities[key].copy()
                self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grads[f'd{key}']
                params[key] += -self.momentum * v_prev + (1 + self.momentum) * self.velocities[key]
            else:
                # Standard momentum update
                self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grads[f'd{key}']
                params[key] += self.velocities[key]
        
        # Update layers with new parameters
        for i, layer in enumerate(layers):
            layer.weights = params[f'W{i+1}']
            layer.bias = params[f'b{i+1}']
            if layer.batch_norm:
                layer.gamma = params[f'gamma{i+1}']
                layer.beta = params[f'beta{i+1}']

class Adam:
    """Adam optimizer implementation.

    This optimizer implements the Adam algorithm as described in 
    'Adam: A Method for Stochastic Optimization' by Kingma and Ba (2014).
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=5.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value  # Maximum allowed gradient value
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
        
    def step(self, layers, X, y, predictions):
        """Update parameters using Adam optimization.
        
        Args:
            layers: List of network layers
            X: Input data
            y: True labels
            predictions: Model predictions
        """
        if not self.m:  # Initialize moment estimates on first update
            for layer in layers:
                if hasattr(layer, 'weights'):
                    self.m[f'weights_{id(layer)}'] = np.zeros_like(layer.weights)
                    self.m[f'bias_{id(layer)}'] = np.zeros_like(layer.bias)
                    self.v[f'weights_{id(layer)}'] = np.zeros_like(layer.weights)
                    self.v[f'bias_{id(layer)}'] = np.zeros_like(layer.bias)
        
        self.t += 1
        
        # Compute gradients
        grad = layers[-1].backward(predictions - y)  # Use MSE derivative
        for i in range(len(layers)-2, -1, -1):
            grad = layers[i].backward(grad)
        
        # Update each layer's parameters
        for layer in layers:
            if hasattr(layer, 'weights'):
                # Get layer gradients
                dw = np.clip(layer.grad_weights, -self.clip_value, self.clip_value)  # Clip gradients
                db = np.clip(layer.grad_bias, -self.clip_value, self.clip_value)     # Clip gradients
                
                # Update moment estimates for weights
                key_w = f'weights_{id(layer)}'
                key_b = f'bias_{id(layer)}'
                
                # Update biased first moment estimates
                self.m[key_w] = self.beta1 * self.m[key_w] + (1 - self.beta1) * dw
                self.m[key_b] = self.beta1 * self.m[key_b] + (1 - self.beta1) * db
                
                # Update biased second raw moment estimates
                self.v[key_w] = self.beta2 * self.v[key_w] + (1 - self.beta2) * np.square(dw)
                self.v[key_b] = self.beta2 * self.v[key_b] + (1 - self.beta2) * np.square(db)
                
                # Compute bias-corrected first moment estimates
                m_hat_w = self.m[key_w] / (1 - self.beta1**self.t)
                m_hat_b = self.m[key_b] / (1 - self.beta1**self.t)
                
                # Compute bias-corrected second raw moment estimates
                v_hat_w = self.v[key_w] / (1 - self.beta2**self.t)
                v_hat_b = self.v[key_b] / (1 - self.beta2**self.t)
                
                # Update parameters
                layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class RMSprop:
    """RMSprop optimizer.
    
    Implements the RMSprop update rule:
    cache = decay_rate * cache + (1 - decay_rate) * gradient^2
    w = w - learning_rate * gradient / sqrt(cache + epsilon)
    
    Args:
        learning_rate: Learning rate (step size) for optimization.
        decay_rate: Decay rate for cache (0 <= decay_rate < 1).
        epsilon: Small value for numerical stability.
        
    Example:
        >>> optimizer = RMSprop(learning_rate=0.001, decay_rate=0.9)
        >>> optimizer.initialize(model.get_parameters())
        >>> for epoch in range(epochs):
        ...     predictions = model.forward(X)
        ...     optimizer.step(model.layers, X, y, predictions)
    """
    
    def __init__(self, learning_rate: float = 0.001, 
                 decay_rate: float = 0.9, 
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def initialize(self, params: Dict[str, np.ndarray]) -> None:
        """Initialize cache for squared gradients.
        
        Args:
            params: Dictionary of parameters to optimize.
        """
        self.cache = {key: np.zeros_like(value) for key, value in params.items()}
    
    def compute_gradients(self, layers: List[Any], X: np.ndarray, 
                         y: np.ndarray, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients using backpropagation.
        
        Args:
            layers: List of network layers.
            X: Input data.
            y: Target values.
            predictions: Network predictions.
            
        Returns:
            Dictionary containing gradients for each parameter.
        """
        m = X.shape[0]
        n_layers = len(layers)
        grads = {}
        
        # Compute output layer error
        error = predictions - y
        
        # Backpropagate through layers
        for i in reversed(range(n_layers)):
            layer = layers[i]
            
            # Compute activation gradient
            if i == n_layers - 1:
                delta = error * layer.activation_derivative(layer.output)
            else:
                delta = np.dot(delta, layers[i+1].weights.T) * layer.activation_derivative(layer.output)
            
            # Compute weight and bias gradients
            grads[f'dW{i+1}'] = np.dot(layer.input.T, delta) / m
            grads[f'db{i+1}'] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Compute batch normalization gradients if applicable
            if layer.batch_norm:
                grads[f'dgamma{i+1}'] = np.sum(delta * layer.normalized_input, axis=0, keepdims=True) / m
                grads[f'dbeta{i+1}'] = np.sum(delta, axis=0, keepdims=True) / m
        
        return grads
    
    def step(self, layers: List[Any], X: np.ndarray, 
             y: np.ndarray, predictions: np.ndarray) -> None:
        """Update parameters using RMSprop.
        
        Args:
            layers: List of network layers.
            X: Input data.
            y: Target values.
            predictions: Network predictions.
        """
        # Get parameter dictionary
        params = {}
        for i, layer in enumerate(layers):
            params[f'W{i+1}'] = layer.weights
            params[f'b{i+1}'] = layer.bias
            if layer.batch_norm:
                params[f'gamma{i+1}'] = layer.gamma
                params[f'beta{i+1}'] = layer.beta
        
        # Initialize cache if needed
        if not self.cache:
            self.initialize(params)
        
        # Compute gradients
        grads = self.compute_gradients(layers, X, y, predictions)
        
        # Update parameters
        for key in params:
            # Update cache
            self.cache[key] = (self.decay_rate * self.cache[key] + 
                             (1 - self.decay_rate) * np.square(grads[f'd{key}']))
            # Update parameters
            params[key] -= (self.learning_rate * grads[f'd{key}']) / (np.sqrt(self.cache[key]) + self.epsilon)
        
        # Update layers with new parameters
        for i, layer in enumerate(layers):
            layer.weights = params[f'W{i+1}']
            layer.bias = params[f'b{i+1}']
            if layer.batch_norm:
                layer.gamma = params[f'gamma{i+1}']
                layer.beta = params[f'beta{i+1}']

# Dictionary mapping optimizer names to classes
OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
    'rmsprop': RMSprop
}
