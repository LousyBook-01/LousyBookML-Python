"""
LousyBookML - Neural Network implementation by LousyBook01
www.youtube.com/@LousyBook01

This module implements a flexible neural network with support for various
activation functions, optimizers, and training configurations.

Features:
- Multiple activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
- Various optimization algorithms (SGD, Momentum, RMSprop)
- Batch normalization and dropout support
- Gradient clipping
- L1/L2 regularization
- Early stopping

Made with ❤️ by LousyBook01
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any
from .layers import LayerConfig

class NeuralNetwork:
    """A flexible neural network implementation supporting various configurations."""
    
    def __init__(self, 
                 layer_configs: List[LayerConfig],
                 learning_rate: float = 0.01,
                 optimizer: str = 'sgd',
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False,
                 l1_lambda: float = 0.0,
                 l2_lambda: float = 0.0):
        """Initialize neural network.
        
        Args:
            layer_configs: List of LayerConfig objects defining the network architecture
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm ('sgd', 'momentum', 'rmsprop')
            dropout_rate: Dropout probability (0.0 means no dropout)
            use_batch_norm: Whether to use batch normalization
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
        """
        self.layer_configs = layer_configs
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        self.weights = []
        self.biases = []
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize network parameters using He initialization."""
        for i in range(len(self.layer_configs) - 1):
            input_dim = self.layer_configs[i].size
            output_dim = self.layer_configs[i + 1].size
            
            # He initialization
            scale = np.sqrt(2.0 / input_dim)
            self.weights.append(np.random.randn(input_dim, output_dim) * scale)
            self.biases.append(np.zeros((1, output_dim)))
    
    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            X: Input data of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout and batch norm)
            
        Returns:
            Network output of shape (batch_size, output_dim)
        """
        self.activations = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            Z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # Batch normalization if enabled
            if self.use_batch_norm and i < len(self.weights) - 1:
                Z = self._batch_normalize(Z, training)
            
            # Activation
            activation_fn = self.layer_configs[i + 1].activation
            A = self._activate(Z, activation_fn)
            
            # Dropout if enabled and in training mode
            if training and self.dropout_rate > 0 and i < len(self.weights) - 1:
                mask = np.random.binomial(1, 1 - self.dropout_rate, A.shape)
                A *= mask / (1 - self.dropout_rate)
            
            current_input = A
            self.activations.append(A)
        
        return current_input
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Backward pass to update network parameters.
        
        Args:
            X: Input data
            y: Target values
        """
        m = X.shape[0]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        dZ = self.activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            dW[i] = np.dot(self.activations[i].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self._activate_derivative(self.activations[i], 
                                                  self.layer_configs[i].activation)
        
        # Update parameters using optimizer
        self._update_parameters(dW, db)
    
    def train(self, 
             X: np.ndarray, 
             y: np.ndarray, 
             epochs: int = 1000,
             batch_size: int = 32,
             validation_data: Optional[tuple] = None,
             early_stopping_patience: int = 0) -> Dict[str, List[float]]:
        """Train the neural network.
        
        Args:
            X: Training data
            y: Target values
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_data: Tuple of (X_val, y_val) for validation
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary containing training history
        """
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward and backward passes
                self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch)
            
            # Calculate losses
            train_loss = self._compute_loss(X, y)
            history['loss'].append(train_loss)
            
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self._compute_loss(X_val, y_val)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if early_stopping_patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            break
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data.
        
        Args:
            X: Input data
            
        Returns:
            Network predictions
        """
        return self.forward(X, training=False)
    
    def _activate(self, Z: np.ndarray, activation_fn: str) -> np.ndarray:
        """Apply activation function."""
        if activation_fn == 'relu':
            return np.maximum(0, Z)
        elif activation_fn == 'leaky_relu':
            return np.where(Z > 0, Z, 0.01 * Z)
        elif activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif activation_fn == 'tanh':
            return np.tanh(Z)
        return Z
    
    def _activate_derivative(self, A: np.ndarray, activation_fn: str) -> np.ndarray:
        """Compute activation function derivative."""
        if activation_fn == 'relu':
            return (A > 0).astype(float)
        elif activation_fn == 'leaky_relu':
            return np.where(A > 0, 1, 0.01)
        elif activation_fn == 'sigmoid':
            return A * (1 - A)
        elif activation_fn == 'tanh':
            return 1 - A ** 2
        return np.ones_like(A)
    
    def _batch_normalize(self, Z: np.ndarray, training: bool) -> np.ndarray:
        """Apply batch normalization."""
        if training:
            mean = np.mean(Z, axis=0, keepdims=True)
            var = np.var(Z, axis=0, keepdims=True) + 1e-8
            Z_norm = (Z - mean) / np.sqrt(var)
        else:
            Z_norm = Z  # Use running statistics in inference (simplified here)
        return Z_norm
    
    def _update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """Update parameters using the specified optimizer."""
        if not hasattr(self, 'momentum_W'):
            self.momentum_W = [np.zeros_like(w) for w in self.weights]
            self.momentum_b = [np.zeros_like(b) for b in self.biases]
            self.rmsprop_W = [np.zeros_like(w) for w in self.weights]
            self.rmsprop_b = [np.zeros_like(b) for b in self.biases]
        
        beta = 0.9  # Momentum/RMSprop parameter
        
        for i in range(len(self.weights)):
            if self.optimizer == 'momentum':
                self.momentum_W[i] = beta * self.momentum_W[i] + (1 - beta) * dW[i]
                self.momentum_b[i] = beta * self.momentum_b[i] + (1 - beta) * db[i]
                dW[i] = self.momentum_W[i]
                db[i] = self.momentum_b[i]
            elif self.optimizer == 'rmsprop':
                self.rmsprop_W[i] = beta * self.rmsprop_W[i] + (1 - beta) * dW[i]**2
                self.rmsprop_b[i] = beta * self.rmsprop_b[i] + (1 - beta) * db[i]**2
                dW[i] = dW[i] / (np.sqrt(self.rmsprop_W[i]) + 1e-8)
                db[i] = db[i] / (np.sqrt(self.rmsprop_b[i]) + 1e-8)
            
            # Add regularization gradients
            if self.l1_lambda > 0:
                dW[i] += self.l1_lambda * np.sign(self.weights[i])
            if self.l2_lambda > 0:
                dW[i] += self.l2_lambda * self.weights[i]
            
            # Update parameters
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the loss with regularization."""
        predictions = self.predict(X)
        m = X.shape[0]
        
        # Mean squared error
        mse_loss = np.mean((predictions - y) ** 2)
        
        # Add regularization loss
        reg_loss = 0
        if self.l1_lambda > 0:
            reg_loss += self.l1_lambda * sum(np.sum(np.abs(w)) for w in self.weights)
        if self.l2_lambda > 0:
            reg_loss += self.l2_lambda * sum(np.sum(w ** 2) for w in self.weights)
        
        return mse_loss + reg_loss
