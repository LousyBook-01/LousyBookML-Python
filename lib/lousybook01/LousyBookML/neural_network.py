"""
Neural Network Implementation from scratch by LousyBook01.

A flexible and feature-rich neural network implementation that supports various
modern deep learning techniques and optimizations.

Features:
    - Multiple activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
    - Various optimization algorithms (SGD, Momentum, RMSprop)
    - Batch normalization support
    - Dropout regularization
    - Customizable layer configurations
    - Gradient monitoring and clipping
    - Weight initialization using He/Xavier initialization
    - Learning rate scheduling
    - L1/L2 regularization
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class LayerConfig:
    """Configuration for a neural network layer.
    
    This class defines the structure and behavior of each layer in the neural network.
    It supports various activation functions, initialization schemes, and regularization
    techniques.
    
    Args:
        input_dim (int): Input dimension of the layer
        output_dim (int): Output dimension of the layer
        activation (str, optional): Activation function name. Defaults to 'relu'.
            Supported values: 'relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear'
        initialization (str, optional): Weight initialization scheme. Defaults to 'he'.
            Supported values: 'he', 'xavier'
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0.
            Must be between 0 and 1.
        l1_reg (float, optional): L1 regularization strength. Defaults to 0.0.
        l2_reg (float, optional): L2 regularization strength. Defaults to 0.0.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
    """
    input_dim: int
    output_dim: int
    activation: str = 'relu'
    initialization: str = 'he'
    dropout_rate: float = 0.0
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    batch_norm: bool = False

    def __post_init__(self):
        """Validate layer configuration."""
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("Layer dimensions must be positive")
        
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        
        if self.l1_reg < 0:
            raise ValueError("L1 regularization strength must be non-negative")
        
        if self.l2_reg < 0:
            raise ValueError("L2 regularization strength must be non-negative")
        
        valid_activations = {'relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear'}
        if self.activation not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}")
        
        valid_inits = {'he', 'xavier'}
        if self.initialization not in valid_inits:
            raise ValueError(f"Initialization must be one of {valid_inits}")

class NeuralNetwork:
    """Enhanced Neural Network with per-layer customization.
    
    A flexible neural network implementation that supports modern deep learning
    features and allows fine-grained control over each layer's configuration.
    
    Args:
        layer_configs (List[LayerConfig]): List of layer configurations defining
            the network architecture
        optimizer (str, optional): Optimization algorithm. Defaults to 'adam'.
            Supported values: 'sgd', 'momentum', 'rmsprop', 'adam'
        learning_rate (float, optional): Initial learning rate. Defaults to 0.001.
        loss (str, optional): Loss function. Defaults to 'mse'.
        gradient_clip (Optional[float], optional): Gradient clipping value. Defaults to None.
    
    Attributes:
        weights (List[np.ndarray]): Layer weights
        biases (List[np.ndarray]): Layer biases
        layer_configs (List[LayerConfig]): Layer configurations
        optimizer (str): Current optimizer
        learning_rate (float): Current learning rate
        gradient_norms (List[float]): History of gradient norms
        active_neurons (List[float]): History of active neuron percentages
    """
    
    SUPPORTED_ACTIVATIONS = {
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, 0.02 * x),
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        'tanh': lambda x: np.tanh(x),
        'linear': lambda x: x
    }
    
    ACTIVATION_DERIVATIVES = {
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky_relu': lambda x: np.where(x > 0, 1, 0.02),
        'sigmoid': lambda x, y: y * (1 - y),  # y is sigmoid(x)
        'tanh': lambda x: 1 - np.tanh(x)**2,
        'linear': lambda x: np.ones_like(x)
    }
    
    INITIALIZATION_SCHEMES = {
        'he': lambda fan_in, fan_out: np.sqrt(2.0 / fan_in),
        'xavier': lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out)),
        'lecun': lambda fan_in, fan_out: np.sqrt(1.0 / fan_in),
        'orthogonal': lambda fan_in, fan_out: 1.0
    }

    def __init__(self, 
                 layer_configs: List[LayerConfig],
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 loss: str = 'mse',
                 gradient_clip: Optional[float] = None):
        """Initialize neural network with given layer configurations."""
        self.layer_configs = layer_configs
        self.num_layers = len(layer_configs)
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.loss = loss.lower()
        self.gradient_clip = gradient_clip
        self.training = True  # Add training mode flag
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for layer_config in layer_configs:
            input_dim = layer_config.input_dim
            output_dim = layer_config.output_dim
            
            # Initialize weights based on initialization method
            if layer_config.initialization == 'xavier':
                scale = np.sqrt(2.0 / (input_dim + output_dim))
                weights = np.random.normal(0, scale, (output_dim, input_dim))
            elif layer_config.initialization == 'he':
                scale = np.sqrt(2.0 / input_dim)
                weights = np.random.normal(0, scale, (output_dim, input_dim))
            else:  # Default to xavier
                scale = np.sqrt(2.0 / (input_dim + output_dim))
                weights = np.random.normal(0, scale, (output_dim, input_dim))
            
            # Initialize biases
            biases = np.zeros((output_dim, 1))
            
            self.weights.append(weights)
            self.biases.append(biases)
        
        # Monitoring metrics
        self.loss_curve_ = []
        self.gradient_norms = []  # Track gradient norms
        self.active_neurons = []  # Track active neurons for ReLU layers
        self.batch_norm_params = {}  # Store batch norm parameters

    def _forward_activation(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Apply forward activation function with improved leaky_relu."""
        if activation == 'sigmoid':
            # Clip values for numerical stability
            z_clipped = np.clip(z, -88.0, 88.0)
            return 1.0 / (1.0 + np.exp(-z_clipped))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'leaky_relu':
            # Improved leaky_relu with adaptive slope
            slope = 0.02  # Slightly larger slope for better gradient flow
            return np.where(z > 0, z, slope * z)
        else:  # linear
            return z

    def _backward_activation(self, z: np.ndarray, activation: str, 
                           activation_output: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute activation function derivative with improved leaky_relu."""
        if activation == 'sigmoid':
            sig = self._forward_activation(z, 'sigmoid')
            return sig * (1 - sig)
        elif activation == 'tanh':
            return 1.0 - np.tanh(z) ** 2
        elif activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif activation == 'leaky_relu':
            # Improved leaky_relu derivative with adaptive slope
            slope = 0.02  # Same slope as forward pass
            return np.where(z > 0, 1, slope)
        else:  # linear
            return np.ones_like(z)

    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Forward propagation step.
        
        Args:
            X: Input data of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Tuple of:
            - activations: List of layer activations
            - cache: List of cached values for backpropagation
        """
        # Ensure input is properly shaped (batch_size, input_dim)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) > 2:
            raise ValueError(f"Input shape {X.shape} is not supported. Expected (batch_size, input_dim) or (input_dim,)")
            
        activations = [X]  # List to store activations
        cache = []  # List to store cache for backpropagation
        
        for i, (layer_config, weights, biases) in enumerate(zip(self.layer_configs, self.weights, self.biases)):
            # Get previous activation
            A_prev = activations[-1]
            
            # Validate input dimensions
            if A_prev.shape[1] != weights.shape[1]:
                raise ValueError(
                    f"Layer {i}: Input dimension mismatch. "
                    f"Got {A_prev.shape[1]}, expected {weights.shape[1]}. "
                    f"Layer config: input_dim={layer_config.input_dim}, output_dim={layer_config.output_dim}"
                )
            
            # Linear forward: Z = XW + b, shape: (batch_size, output_dim)
            Z = np.dot(A_prev, weights.T) + biases.T
            
            # Apply batch normalization if enabled
            if layer_config.batch_norm:
                # Initialize batch norm parameters if not exists
                if i not in self.batch_norm_params:
                    self.batch_norm_params[i] = {
                        'gamma': np.ones((1, weights.shape[0])),  # (1, output_dim)
                        'beta': np.zeros((1, weights.shape[0])),  # (1, output_dim)
                        'moving_mean': np.zeros((1, weights.shape[0])),  # (1, output_dim)
                        'moving_var': np.ones((1, weights.shape[0])),  # (1, output_dim)
                        'epsilon': 1e-8
                    }
                
                # Apply batch normalization
                if self.training:
                    batch_mean = np.mean(Z, axis=0, keepdims=True)  # (1, output_dim)
                    batch_var = np.var(Z, axis=0, keepdims=True)  # (1, output_dim)
                    
                    # Normalize
                    Z_norm = (Z - batch_mean) / np.sqrt(batch_var + self.batch_norm_params[i]['epsilon'])
                    
                    # Update moving averages
                    momentum = 0.9
                    self.batch_norm_params[i]['moving_mean'] = (
                        momentum * self.batch_norm_params[i]['moving_mean'] +
                        (1 - momentum) * batch_mean
                    )
                    self.batch_norm_params[i]['moving_var'] = (
                        momentum * self.batch_norm_params[i]['moving_var'] +
                        (1 - momentum) * batch_var
                    )
                else:
                    # Use moving averages for inference
                    Z_norm = (
                        (Z - self.batch_norm_params[i]['moving_mean']) /
                        np.sqrt(self.batch_norm_params[i]['moving_var'] + self.batch_norm_params[i]['epsilon'])
                    )
                
                # Scale and shift
                Z = (
                    self.batch_norm_params[i]['gamma'] * Z_norm +
                    self.batch_norm_params[i]['beta']
                )
                
                # Cache batch norm params for backward pass
                cache.append((
                    A_prev, weights, biases, Z,
                    {'norm': Z_norm, 'mean': batch_mean, 'var': batch_var} if self.training else None
                ))
            else:
                cache.append((A_prev, weights, biases, Z))
            
            # Activation
            A = self._forward_activation(Z, layer_config.activation)
            
            # Apply dropout during training if enabled
            if self.training and layer_config.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - layer_config.dropout_rate, A.shape)
                A *= mask / (1 - layer_config.dropout_rate)  # Scale to maintain expected values
                cache[-1] = cache[-1] + (mask,)  # Add mask to cache
            
            activations.append(A)
        
        return activations, cache

    def _clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradients to prevent exploding gradients."""
        if self.gradient_clip is None:
            return gradients
            
        # Compute global norm
        global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients))
        self.gradient_norms.append(float(global_norm))
        
        if global_norm > self.gradient_clip:
            scale = self.gradient_clip / (global_norm + 1e-8)  # Fixed epsilon reference
            return [g * scale for g in gradients]
        return gradients

    def backward_propagation(self, X: np.ndarray, y: np.ndarray,
                           activations: List[np.ndarray],
                           cache: List[Tuple], sample_weights: Optional[np.ndarray] = None) -> Dict[str, List[np.ndarray]]:
        """Backward propagation with gradient monitoring and clipping."""
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = (activations[-1] - y)
        if self.layer_configs[-1].activation != 'linear':
            delta *= self._backward_activation(cache[-1][3], self.layer_configs[-1].activation)
        
        if sample_weights is not None:
            delta *= sample_weights[:, np.newaxis]
        
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True).T
        nabla_w[-1] = np.dot(delta.T, activations[-2])

        # Hidden layers
        for l in range(2, self.num_layers + 1):
            z = cache[-l][3]
            delta = np.dot(delta, self.weights[-l+1])
            if self.layer_configs[-l].activation != 'linear':
                delta *= self._backward_activation(z, self.layer_configs[-l].activation)
            
            if self.training and self.layer_configs[-l].dropout_rate > 0:
                delta *= cache[-l][4] / (1 - self.layer_configs[-l].dropout_rate)
            
            if sample_weights is not None:
                delta *= sample_weights[:, np.newaxis]
            
            nabla_b[-l] = np.sum(delta, axis=0, keepdims=True).T
            nabla_w[-l] = np.dot(delta.T, activations[-l-1])

            # Add L1/L2 regularization gradients
            if self.layer_configs[-l].l1_reg > 0:
                nabla_w[-l] += self.layer_configs[-l].l1_reg * np.sign(self.weights[-l])
            if self.layer_configs[-l].l2_reg > 0:
                nabla_w[-l] += self.layer_configs[-l].l2_reg * self.weights[-l]

        return {'weights': nabla_w, 'biases': nabla_b}

    def _update_parameters(self, gradients: Dict[str, List[np.ndarray]], epoch: int) -> None:
        """Update network parameters using the chosen optimizer."""
        if not hasattr(self, 'optimizer_state'):
            self.optimizer_state = {
                'momentum': {'weights': [np.zeros_like(w) for w in self.weights],
                           'biases': [np.zeros_like(b) for b in self.biases]},
                'rmsprop': {'weights': [np.zeros_like(w) for w in self.weights],
                           'biases': [np.zeros_like(b) for b in self.biases]},
                'adam': {
                    'm_weights': [np.zeros_like(w) for w in self.weights],
                    'm_biases': [np.zeros_like(b) for b in self.biases],
                    'v_weights': [np.zeros_like(w) for w in self.weights],
                    'v_biases': [np.zeros_like(b) for b in self.biases],
                    't': 0
                }
            }

        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients['weights'][i]
                self.biases[i] -= self.learning_rate * gradients['biases'][i]

        elif self.optimizer == 'momentum':
            beta = 0.9
            for i in range(len(self.weights)):
                # Update weights
                self.optimizer_state['momentum']['weights'][i] = (
                    beta * self.optimizer_state['momentum']['weights'][i] +
                    (1 - beta) * gradients['weights'][i]
                )
                self.weights[i] -= self.learning_rate * self.optimizer_state['momentum']['weights'][i]
                
                # Update biases
                self.optimizer_state['momentum']['biases'][i] = (
                    beta * self.optimizer_state['momentum']['biases'][i] +
                    (1 - beta) * gradients['biases'][i]
                )
                self.biases[i] -= self.learning_rate * self.optimizer_state['momentum']['biases'][i]

        elif self.optimizer == 'rmsprop':
            beta = 0.999
            epsilon = 1e-8
            for i in range(len(self.weights)):
                # Update weights
                self.optimizer_state['rmsprop']['weights'][i] = (
                    beta * self.optimizer_state['rmsprop']['weights'][i] +
                    (1 - beta) * np.square(gradients['weights'][i])
                )
                self.weights[i] -= (self.learning_rate * gradients['weights'][i] /
                                  (np.sqrt(self.optimizer_state['rmsprop']['weights'][i] + epsilon)))
                
                # Update biases
                self.optimizer_state['rmsprop']['biases'][i] = (
                    beta * self.optimizer_state['rmsprop']['biases'][i] +
                    (1 - beta) * np.square(gradients['biases'][i])
                )
                self.biases[i] -= (self.learning_rate * gradients['biases'][i] /
                                 (np.sqrt(self.optimizer_state['rmsprop']['biases'][i] + epsilon)))

        elif self.optimizer == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            self.optimizer_state['adam']['t'] += 1
            t = self.optimizer_state['adam']['t']
            
            for i in range(len(self.weights)):
                # Update weights
                self.optimizer_state['adam']['m_weights'][i] = (
                    beta1 * self.optimizer_state['adam']['m_weights'][i] +
                    (1 - beta1) * gradients['weights'][i]
                )
                self.optimizer_state['adam']['v_weights'][i] = (
                    beta2 * self.optimizer_state['adam']['v_weights'][i] +
                    (1 - beta2) * np.square(gradients['weights'][i])
                )
                
                m_hat = self.optimizer_state['adam']['m_weights'][i] / (1 - beta1**t)
                v_hat = self.optimizer_state['adam']['v_weights'][i] / (1 - beta2**t)
                
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                
                # Update biases
                self.optimizer_state['adam']['m_biases'][i] = (
                    beta1 * self.optimizer_state['adam']['m_biases'][i] +
                    (1 - beta1) * gradients['biases'][i]
                )
                self.optimizer_state['adam']['v_biases'][i] = (
                    beta2 * self.optimizer_state['adam']['v_biases'][i] +
                    (1 - beta2) * np.square(gradients['biases'][i])
                )
                
                m_hat = self.optimizer_state['adam']['m_biases'][i] / (1 - beta1**t)
                v_hat = self.optimizer_state['adam']['v_biases'][i] / (1 - beta2**t)
                
                self.biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        """Compute loss between predictions and true values.
        
        Args:
            y_pred: Predicted values of shape (batch_size, output_dim)
            y_true: True values of shape (batch_size, output_dim)
            sample_weights: Optional array of sample weights for importance sampling
            
        Returns:
            Loss value as float
        """
        if self.loss == 'mse':
            squared_errors = (y_pred - y_true) ** 2
            if sample_weights is not None:
                # Reshape sample weights to match squared errors shape
                weights = sample_weights.reshape(-1, 1)
                return np.mean(weights * squared_errors)
            else:
                return np.mean(squared_errors)
        elif self.loss == 'binary_crossentropy':
            # Clip predictions to prevent log(0)
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            if sample_weights is not None:
                weights = sample_weights.reshape(-1, 1)
                return -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
            else:
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss == 'categorical_crossentropy':
            # Clip predictions to prevent log(0)
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            if sample_weights is not None:
                weights = sample_weights.reshape(-1, 1)
                return -np.mean(weights * np.sum(y_true * np.log(y_pred), axis=1, keepdims=True))
            else:
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def _check_early_stopping(self, val_loss: float, patience: int = 5, min_delta: float = 1e-4) -> bool:
        """Check if training should stop based on validation loss.
        
        Args:
            val_loss: Current validation loss
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in loss to qualify as an improvement
            
        Returns:
            True if training should stop, False otherwise
        """
        if not hasattr(self, 'val_losses_'):
            self.val_losses_ = []
        
        self.val_losses_.append(val_loss)
        
        if len(self.val_losses_) > patience:
            # Check if loss hasn't improved for 'patience' epochs
            best_loss = min(self.val_losses_[:-patience])
            if val_loss > best_loss - min_delta:
                return True
        return False

    def save_model(self, filepath: str) -> None:
        """Save model parameters to file.
        
        Args:
            filepath: Path to save the model (should end with .npz)
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        model_state = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_configs': self.layer_configs,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'gradient_clip': self.gradient_clip,
            'optimizer_state': self.optimizer_state if hasattr(self, 'optimizer_state') else None,
            'batch_norm_params': self.batch_norm_params
        }
        np.savez_compressed(filepath, **{'model_state': model_state})
        
    def load_model(self, filepath: str) -> None:
        """Load model parameters from file.
        
        Args:
            filepath: Path to the saved model file
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        with np.load(filepath, allow_pickle=True) as data:
            model_state = data['model_state'].item()
            
        self.weights = model_state['weights']
        self.biases = model_state['biases']
        self.layer_configs = model_state['layer_configs']
        self.optimizer = model_state['optimizer']
        self.learning_rate = model_state['learning_rate']
        self.loss = model_state['loss']
        self.gradient_clip = model_state['gradient_clip']
        self.batch_norm_params = model_state['batch_norm_params']
        
        if model_state['optimizer_state'] is not None:
            self.optimizer_state = model_state['optimizer_state']

    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              max_iter: int = 100, batch_size: int = 32, patience: int = 5, sample_weight: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Train the neural network with early stopping and batch training.
        
        Args:
            X: Training data of shape (n_samples, input_dim)
            y: Target values of shape (n_samples, output_dim)
            validation_data: Optional tuple of (X_val, y_val) for validation
            max_iter: Maximum number of epochs
            batch_size: Size of mini-batches
            patience: Number of epochs to wait before early stopping
            sample_weight: Optional array of sample weights for importance sampling
            
        Returns:
            Dictionary containing training history
        """
        self.training = True
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}
        
        try:
            for epoch in range(max_iter):
                # Mini-batch training
                indices = np.random.permutation(n_samples)
                epoch_losses = []
                
                for start_idx in range(0, n_samples, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Get batch weights if provided
                    batch_weights = None
                    if sample_weight is not None:
                        batch_weights = sample_weight[batch_indices]
                    
                    # Forward pass
                    activations, cache = self.forward_propagation(X_batch)
                    
                    # Compute loss with sample weights if provided
                    if batch_weights is not None:
                        batch_loss = self._compute_loss(activations[-1], y_batch, batch_weights)
                    else:
                        batch_loss = self._compute_loss(activations[-1], y_batch)
                    epoch_losses.append(batch_loss)
                    
                    # Backward pass with sample weights
                    gradients = self.backward_propagation(X_batch, y_batch, activations, cache, batch_weights)
                    
                    # Update parameters
                    self._update_parameters(gradients, epoch)
                
                # Compute epoch metrics
                epoch_loss = np.mean(epoch_losses)
                history['loss'].append(epoch_loss)
                
                # Validation step
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_activations, _ = self.forward_propagation(X_val)
                    val_loss = self._compute_loss(val_activations[-1], y_val)
                    history['val_loss'].append(val_loss)
                    
                    # Check early stopping
                    if self._check_early_stopping(val_loss, patience=patience):
                        break
        
        finally:
            self.training = False
        
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data.
        
        Args:
            X: Input data of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        # Temporarily disable training mode for prediction
        training_mode = self.training
        self.training = False
        
        try:
            # Forward pass
            activations, _ = self.forward_propagation(X)
            predictions = activations[-1]
            
            # Ensure output has correct shape (batch_size, output_dim)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)
                
            return predictions
        finally:
            # Restore training mode
            self.training = training_mode