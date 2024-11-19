"""
Neural Network Implementation from scratch by LousyBook01.

Features:
    - Multiple activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
    - Various optimization algorithms (SGD, Momentum, RMSprop)
    - Batch normalization support
    - Dropout regularization
    - Customizable layer configurations
    - Gradient monitoring and clipping
    - Weight initialization using He/Xavier initialization
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    size: int
    activation: str = 'relu'
    initialization: str = 'he'
    dropout_rate: float = 0.0
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    bias_init: str = 'zeros'
    weight_scale: float = 1.0

class NeuralNetwork:
    """Enhanced Neural Network with per-layer customization."""
    
    SUPPORTED_ACTIVATIONS = {
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        'tanh': lambda x: np.tanh(x),
        'linear': lambda x: x
    }
    
    ACTIVATION_DERIVATIVES = {
        'relu': lambda x: np.where(x > 0, 1, 0),
        'leaky_relu': lambda x: np.where(x > 0, 1, 0.01),
        'sigmoid': lambda x, y: y * (1 - y),  # y is sigmoid(x)
        'tanh': lambda x: 1 - np.tanh(x)**2,
        'linear': lambda x: np.ones_like(x)
    }
    
    INITIALIZATION_SCHEMES = {
        'he': lambda fan_in, fan_out: np.sqrt(2.0 / fan_in),
        'xavier': lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out)),
        'lecun': lambda fan_in, fan_out: np.sqrt(1.0 / fan_in),
        'orthogonal': lambda fan_in, fan_out: 1.0  # Scale will be applied after orthogonalization
    }

    def __init__(self, 
                 layer_configs: List[LayerConfig],
                 optimizer: str = 'momentum',
                 learning_rate: float = 0.01,
                 momentum_beta: float = 0.9,
                 rmsprop_beta: float = 0.999,
                 epsilon: float = 1e-8,
                 gradient_clip: float = 5.0,
                 batch_norm: bool = False):
        """Initialize the neural network with layer-specific configurations."""
        self.layer_configs = layer_configs
        self.num_layers = len(layer_configs)
        self.layer_sizes = [config.size for config in layer_configs]
        
        # Validate layer sizes
        for i, size in enumerate(self.layer_sizes):
            if size <= 0:
                raise ValueError(f"Layer size must be positive, got {size} for layer {i}")
        
        # Optimization parameters
        self.optimizer = optimizer.lower()
        if self.optimizer not in ['sgd', 'momentum', 'rmsprop']:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.learning_rate = learning_rate
        self.momentum_beta = momentum_beta
        self.rmsprop_beta = rmsprop_beta
        self.epsilon = epsilon
        self.gradient_clip = gradient_clip
        self.batch_norm = batch_norm
        
        # Initialize network parameters
        self.weights = []
        self.biases = []
        self.velocity_w = []  # For momentum
        self.velocity_b = []
        self.cache_w = []    # For RMSprop
        self.cache_b = []
        
        # Batch normalization parameters
        if batch_norm:
            self.gamma = []   # Scale parameter
            self.beta = []    # Shift parameter
            self.running_mean = []
            self.running_var = []
        
        # Monitoring metrics
        self.active_neurons = []
        self.gradient_norms = []
        
        self._init_parameters()

    def _init_parameters(self):
        """Initialize network parameters using layer-specific configurations."""
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            config = self.layer_configs[i + 1]
            
            # Get initialization scale based on initialization scheme
            if config.initialization == 'he':
                scale = np.sqrt(2.0 / fan_in)
            elif config.initialization == 'xavier':
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            elif config.initialization == 'orthogonal':
                scale = 1.0
            else:
                scale = np.sqrt(1.0 / fan_in)  # Default to LeCun initialization
            
            # Initialize weights with improved scaling
            if config.initialization == 'orthogonal':
                rng = np.random.RandomState(i)
                W = rng.randn(fan_out, fan_in)
                u, _, vh = np.linalg.svd(W, full_matrices=False)
                w = u @ vh * scale * config.weight_scale
            else:
                w = np.random.randn(fan_out, fan_in) * scale * config.weight_scale
            
            # Initialize biases with optimized values for different activations
            if config.activation in ['relu', 'leaky_relu']:
                b = np.ones((fan_out, 1)) * 0.01  # Small positive bias for ReLU
            elif config.activation == 'tanh':
                b = np.zeros((fan_out, 1))  # Zero initialization for tanh
            else:
                b = np.random.randn(fan_out, 1) * 0.01  # Small random for others
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Initialize optimizer states
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))
            self.cache_w.append(np.zeros_like(w))
            self.cache_b.append(np.zeros_like(b))
            
            if self.batch_norm:
                self.gamma.append(np.ones((fan_out, 1)))
                self.beta.append(np.zeros((fan_out, 1)))
                self.running_mean.append(np.zeros((fan_out, 1)))
                self.running_var.append(np.ones((fan_out, 1)))

    def _forward_activation(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Apply forward activation function."""
        return self.SUPPORTED_ACTIVATIONS[activation](z)

    def _backward_activation(self, z: np.ndarray, activation: str, 
                           activation_output: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute derivative of activation function."""
        if activation == 'sigmoid':
            return self.ACTIVATION_DERIVATIVES[activation](z, activation_output)
        return self.ACTIVATION_DERIVATIVES[activation](z)

    def forward_propagation(self, x: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward propagation with layer-specific activations and batch norm."""
        activations = [x]
        zs = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            
            # Apply batch normalization if enabled
            if self.batch_norm and i < self.num_layers - 2:  # Don't apply to output layer
                if training:
                    batch_mean = np.mean(z, axis=1, keepdims=True)
                    batch_var = np.var(z, axis=1, keepdims=True) + self.epsilon
                    z_norm = (z - batch_mean) / np.sqrt(batch_var)
                    
                    # Update running statistics
                    self.running_mean[i] = 0.9 * self.running_mean[i] + 0.1 * batch_mean
                    self.running_var[i] = 0.9 * self.running_var[i] + 0.1 * batch_var
                else:
                    z_norm = (z - self.running_mean[i]) / np.sqrt(self.running_var[i] + self.epsilon)
                
                z = self.gamma[i] * z_norm + self.beta[i]
            
            zs.append(z)
            activation = self._forward_activation(z, self.layer_configs[i + 1].activation)
            
            # Apply dropout during training if specified
            if training and self.layer_configs[i + 1].dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.layer_configs[i + 1].dropout_rate, 
                                        size=activation.shape) / (1 - self.layer_configs[i + 1].dropout_rate)
                activation = activation * mask
            
            activations.append(activation)
            
            # Monitor active neurons for ReLU-like functions
            if self.layer_configs[i + 1].activation in ['relu', 'leaky_relu']:
                active_percent = np.mean(activation > 0) * 100
                self.active_neurons.append(active_percent)
        
        return activations, zs

    def _clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradients to prevent exploding gradients."""
        clipped = []
        for grad in gradients:
            norm = np.linalg.norm(grad)
            self.gradient_norms.append(norm)
            if norm > self.gradient_clip:
                grad = grad * (self.gradient_clip / norm)
            clipped.append(grad)
        return clipped

    def _get_learning_rate(self, base_lr: float, iteration: int) -> float:
        """Implement learning rate scheduling with improved warmup."""
        warmup = 200  # Extended warmup period
        if iteration < warmup:
            # Cosine warmup for smoother transition
            progress = iteration / warmup
            return base_lr * (1 - np.cos(progress * np.pi)) / 2
        else:
            # Cosine decay with restarts
            decay_steps = 500
            num_cycles = (iteration - warmup) // decay_steps
            progress = ((iteration - warmup) % decay_steps) / decay_steps
            
            # Reduce learning rate after each restart
            cycle_factor = 0.8 ** num_cycles
            min_lr = base_lr * 0.01
            
            current_progress = progress + num_cycles
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr + (base_lr * cycle_factor - min_lr) * cosine_decay

    def _update_parameters(self, nabla_w: List[np.ndarray],
                         nabla_b: List[np.ndarray],
                         learning_rate: float,
                         iteration: int):
        """Update network parameters with improved momentum implementation."""
        # Get scheduled learning rate
        effective_lr = self._get_learning_rate(learning_rate, iteration)
        
        # Dynamic learning rate adjustment based on layer depth
        layer_multipliers = [1.0 / (i + 1) for i in range(len(self.weights))]
        
        # Adjust learning rate based on activation function and gradient magnitude
        activation_multipliers = []
        for config in self.layer_configs[1:]:
            if config.activation == 'relu':
                activation_multipliers.append(1.0)
            elif config.activation == 'leaky_relu':
                activation_multipliers.append(1.2)
            elif config.activation == 'tanh':
                activation_multipliers.append(0.8)
            else:
                activation_multipliers.append(0.5)

        if self.optimizer == 'sgd':
            for i, (w, nw, b, nb) in enumerate(zip(self.weights, nabla_w, self.biases, nabla_b)):
                lr = effective_lr * layer_multipliers[i] * activation_multipliers[i]
                self.weights[i] = w - lr * nw
                self.biases[i] = b - lr * nb

        elif self.optimizer == 'momentum':
            # Adaptive momentum with layer-specific beta
            for i, (w, nw, b, nb) in enumerate(zip(self.weights, nabla_w, self.biases, nabla_b)):
                lr = effective_lr * layer_multipliers[i] * activation_multipliers[i]
                beta = min(self.momentum_beta, 1.0 - 1.0/(iteration + 1))
                beta = beta * (1.0 - 0.1 * i/len(self.weights))  # Reduce momentum for deeper layers
                
                # Update velocities with adaptive momentum
                self.velocity_w[i] = beta * self.velocity_w[i] + lr * nw
                self.velocity_b[i] = beta * self.velocity_b[i] + lr * nb
                
                # Apply updates with bias correction
                correction = 1.0 - beta ** (iteration + 1)
                self.weights[i] = w - self.velocity_w[i] / correction
                self.biases[i] = b - self.velocity_b[i] / correction

        elif self.optimizer == 'rmsprop':
            # Layer-specific beta for RMSprop
            for i, (w, nw, b, nb) in enumerate(zip(self.weights, nabla_w, self.biases, nabla_b)):
                lr = effective_lr * layer_multipliers[i] * activation_multipliers[i]
                beta = min(self.rmsprop_beta, 1.0 - 1.0/(iteration + 1))
                beta = beta * (1.0 - 0.05 * i/len(self.weights))  # Slightly reduce beta for deeper layers
                
                # Update cache with adaptive beta
                self.cache_w[i] = beta * self.cache_w[i] + (1 - beta) * (nw ** 2)
                self.cache_b[i] = beta * self.cache_b[i] + (1 - beta) * (nb ** 2)
                
                # Apply bias correction
                correction = 1.0 - beta ** (iteration + 1)
                
                # Update with improved gradient rescaling
                self.weights[i] = w - (lr * nw) / (np.sqrt(self.cache_w[i] / correction) + self.epsilon)
                self.biases[i] = b - (lr * nb) / (np.sqrt(self.cache_b[i] / correction) + self.epsilon)

    def backward_propagation(self, x: np.ndarray, y: np.ndarray,
                           activations: List[np.ndarray],
                           zs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward propagation with gradient monitoring and clipping."""
        batch_size = x.shape[1]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = (activations[-1] - y) * self._backward_activation(zs[-1], self.layer_configs[-1].activation, activations[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / batch_size
        nabla_w[-1] = np.dot(delta, activations[-2].T) / batch_size

        # Hidden layers
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * self._backward_activation(zs[-l], self.layer_configs[-l].activation)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / batch_size
            nabla_w[-l] = np.dot(delta, activations[-l-1].T) / batch_size

        # Clip gradients
        nabla_w = self._clip_gradients(nabla_w)
        nabla_b = self._clip_gradients(nabla_b)

        return nabla_w, nabla_b

    def train(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1,
             epochs: int = 100, batch_size: int = 32, verbose: bool = True):
        """Train the neural network with improved monitoring and stability."""
        # Input validation and reshaping
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
            
        if X.shape[0] != self.layer_sizes[0]:
            X = X.T
        if y.shape[0] != self.layer_sizes[-1]:
            y = y.T
            
        if X.shape[1] != y.shape[1]:
            raise ValueError("Number of samples in X and y must match")
            
        n_samples = X.shape[1]
        iteration = 0
        best_loss = float('inf')
        patience = 50  # Early stopping patience
        no_improve = 0
        
        # Reset monitoring lists
        self.gradient_norms = []
        self.active_neurons = []
        
        # Initialize momentum for loss improvement
        loss_momentum = None
        beta_loss = 0.9
        
        for epoch in range(epochs):
            # Shuffle data with reproducible order within epoch
            rng = np.random.RandomState(epoch)
            indices = rng.permutation(n_samples)
            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]
            
            epoch_losses = []
            
            # Mini-batch training with dynamic batch size
            current_batch_size = min(batch_size, max(2, n_samples // (epoch // 100 + 1)))
            
            for start_idx in range(0, n_samples, current_batch_size):
                iteration += 1
                end_idx = min(start_idx + current_batch_size, n_samples)
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Forward pass
                activations, zs = self.forward_propagation(X_batch, training=True)
                
                # Compute batch loss
                batch_loss = np.mean((activations[-1] - y_batch) ** 2)
                epoch_losses.append(batch_loss)
                
                # Backward pass with gradient clipping
                nabla_w, nabla_b = self.backward_propagation(X_batch, y_batch, activations, zs)
                
                # Update parameters with current learning rate
                self._update_parameters(nabla_w, nabla_b, learning_rate, iteration)
            
            # Compute epoch loss with momentum
            epoch_loss = np.mean(epoch_losses)
            if loss_momentum is None:
                loss_momentum = epoch_loss
            else:
                loss_momentum = beta_loss * loss_momentum + (1 - beta_loss) * epoch_loss
            
            # Early stopping check with momentum
            if loss_momentum < best_loss:
                best_loss = loss_momentum
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Compute and print metrics
            if verbose and epoch % 10 == 0:
                activations, _ = self.forward_propagation(X, training=False)
                loss = np.mean((activations[-1] - y) ** 2)
                
                # Compute percentage of active neurons in ReLU layers
                active_neurons = []
                for i, config in enumerate(self.layer_configs[1:], 1):
                    if config.activation in ['relu', 'leaky_relu']:
                        active = np.mean(activations[i] > 0) * 100
                        active_neurons.append(active)
                
                avg_active = np.mean(active_neurons) if active_neurons else 0
                
                # Track gradient norms
                grad_norm = np.mean([np.linalg.norm(nw) for nw in nabla_w])
                self.gradient_norms.append(grad_norm)
                
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Active Neurons: {avg_active:.1f}%, Gradient Norm: {grad_norm:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input data."""
        activations, _ = self.forward_propagation(X)
        return activations[-1]