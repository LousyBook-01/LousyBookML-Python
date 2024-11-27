"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Model Module
This module provides the core neural network implementation:
- Layer class for neural network layers
- NeuralNetwork class for building and training networks
- Support for various layer types and configurations
- Batch normalization capability
- Mini-batch training with optimizers

Example:
    >>> from LousyBookML.neural_network.model import NeuralNetwork
    >>> model = NeuralNetwork([
    ...     {'units': 64, 'activation': 'relu'},
    ...     {'units': 32, 'activation': 'relu'},
    ...     {'units': 10, 'activation': 'softmax'}
    ... ])
    >>> model.fit(X_train, y_train, epochs=100, batch_size=32)
    >>> predictions = model.predict(X_test)
"""

import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
from .activations import ACTIVATION_FUNCTIONS
from .losses import LOSS_FUNCTIONS
from .optimizers import OPTIMIZERS
from .initializers import INITIALIZERS

class Layer:
    """Neural network layer with optional batch normalization.
    
    A fully connected layer that performs:
    output = activation(gamma * normalize(W * x + b) + beta) if batch_norm
    output = activation(W * x + b) otherwise
    
    Args:
        units: Number of neurons in the layer
        activation: Activation function name ('relu', 'sigmoid', etc.)
        kernel_initializer: Weight initialization method
        seed: Random seed for reproducibility
        batch_norm: Whether to use batch normalization
        
    Attributes:
        weights: Weight matrix of shape (input_dim, units)
        bias: Bias vector of shape (1, units)
        gamma: Batch norm scaling parameter
        beta: Batch norm shift parameter
        running_mean: Running mean for batch norm inference
        running_var: Running variance for batch norm inference
        
    Example:
        >>> layer = Layer(64, activation='relu', batch_norm=True)
        >>> layer.initialize(input_shape=128)
        >>> output = layer.forward(input_data)
    """
    
    def __init__(self, units: int, activation: str = 'relu', 
                 kernel_initializer: str = 'xavier_uniform', seed: Optional[int] = None,
                 batch_norm: bool = False):
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.seed = seed
        self.batch_norm = batch_norm
        
        # Get activation function and its derivative
        activation_pair = ACTIVATION_FUNCTIONS[activation]
        self.activation_fn = activation_pair[0]  # Forward function
        self.activation_derivative = activation_pair[1]  # Backward function
        
        self.initializer = INITIALIZERS[kernel_initializer]
        
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None
        self.normalized_input = None
        self.grad_weights = None
        self.grad_bias = None
        
        # Batch normalization parameters
        if batch_norm:
            self.gamma = None  # Scale parameter
            self.beta = None   # Shift parameter
            self.running_mean = None
            self.running_var = None
            self.epsilon = 1e-8
            self.momentum = 0.99
        
    def initialize(self, input_shape: int):
        self.weights = self.initializer((input_shape, self.units), seed=self.seed)
        self.bias = np.zeros((1, self.units))
        
        if self.batch_norm:
            self.gamma = np.ones((1, self.units))
            self.beta = np.zeros((1, self.units))
            self.running_mean = np.zeros((1, self.units))
            self.running_var = np.ones((1, self.units))
        
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = X
        output = np.dot(X, self.weights) + self.bias
        
        if self.batch_norm:
            if training:
                # Calculate mean and variance for current batch
                batch_mean = np.mean(output, axis=0, keepdims=True)
                batch_var = np.var(output, axis=0, keepdims=True) + self.epsilon
                
                # Update running statistics
                self.running_mean = (self.momentum * self.running_mean + 
                                   (1 - self.momentum) * batch_mean)
                self.running_var = (self.momentum * self.running_var + 
                                  (1 - self.momentum) * batch_var)
                
                # Normalize
                self.normalized_input = (output - batch_mean) / np.sqrt(batch_var)
            else:
                # Use running statistics for inference
                self.normalized_input = ((output - self.running_mean) / 
                            np.sqrt(self.running_var + self.epsilon))
            
            # Scale and shift
            output = self.gamma * self.normalized_input + self.beta
        
        self.output = self.activation_fn(output)
        return self.output
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute gradients using backpropagation.
        
        Args:
            grad: Gradient from previous layer
            
        Returns:
            Gradient for this layer
        """
        if self.batch_norm:
            # Backpropagate through batch norm
            grad_gamma = np.sum(grad * self.normalized_input, axis=0, keepdims=True)
            grad_beta = np.sum(grad, axis=0, keepdims=True)
            
            # Update gradients
            self.gamma -= grad_gamma
            self.beta -= grad_beta
            
            # Backpropagate through normalization
            grad_normalized = grad * self.gamma
            grad_mean = np.sum(grad_normalized * (-1 / np.sqrt(self.running_var + self.epsilon)), axis=0, keepdims=True)
            grad_var = np.sum(grad_normalized * (-0.5) * (self.normalized_input / (self.running_var + self.epsilon)), axis=0, keepdims=True)
            
            # Backpropagate through mean and variance
            grad_input = (grad_normalized / np.sqrt(self.running_var + self.epsilon)) + (grad_mean / self.input.shape[0]) + (2 * grad_var * self.normalized_input / self.input.shape[0])
        else:
            grad_input = grad
        
        # Backpropagate through activation
        grad_activated = grad_input * self.activation_derivative(self.output)
        
        # Compute gradients for weights and bias
        self.grad_weights = np.dot(self.input.T, grad_activated)
        self.grad_bias = np.sum(grad_activated, axis=0, keepdims=True)
        
        # Compute gradient for next layer
        grad_next = np.dot(grad_activated, self.weights.T)
        
        return grad_next

class NeuralNetwork:
    """Neural network model with configurable architecture and training options.
    
    A flexible neural network implementation that supports:
    - Multiple layers with different activations
    - Various optimizers (SGD, Adam)
    - Batch normalization
    - Early stopping
    - Mini-batch training
    - Validation during training
    
    Args:
        architecture: List of layer configurations or Layer objects
        loss: Loss function name ('mean_squared_error', 'categorical_crossentropy', etc.)
        optimizer: Optimizer name ('sgd', 'adam')
        learning_rate: Learning rate for the optimizer
        **optimizer_params: Additional parameters for the optimizer
        
    Attributes:
        layers: List of Layer objects forming the network
        loss_fn: Loss function for training
        optimizer: Optimizer instance for updating weights
        
    Example:
        >>> # Create a simple classifier
        >>> model = NeuralNetwork([
        ...     {'units': 128, 'activation': 'relu'},
        ...     {'units': 10, 'activation': 'softmax'}
        ... ], loss='categorical_crossentropy', optimizer='adam')
        >>> 
        >>> # Train the model
        >>> history = model.fit(X_train, y_train, 
        ...                    epochs=100, 
        ...                    batch_size=32,
        ...                    validation_data=(X_val, y_val))
    """
    
    def __init__(self, 
                 architecture: Union[List[Dict[str, Union[int, str]]], List[Layer]],
                 loss: str = 'mean_squared_error',
                 optimizer: str = 'sgd',
                 learning_rate: float = 0.01,
                 **optimizer_params):
        self.layers = []
        
        # Process architecture
        for layer_config in architecture:
            if isinstance(layer_config, Layer):
                self.layers.append(layer_config)
            elif isinstance(layer_config, dict):
                self.layers.append(Layer(**layer_config))
            else:
                raise ValueError("Layer configuration must be a Layer object or dictionary")
        
        # Set up loss function
        if loss not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function: {loss}")
        self.loss_name = loss
        self.loss_fn = LOSS_FUNCTIONS[loss][0]  # Get loss function
        self.loss_derivative = LOSS_FUNCTIONS[loss][1]  # Get derivative
        
        # Set up optimizer
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        optimizer_class = OPTIMIZERS[optimizer]
        self.optimizer = optimizer_class(learning_rate=learning_rate, **optimizer_params)
        
    def initialize(self, input_shape: int):
        """Initialize network weights.
        
        Args:
            input_shape: Number of input features
        """
        current_shape = input_shape
        for layer in self.layers:
            layer.initialize(current_shape)
            current_shape = layer.units
            
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            X: Input data
            
        Returns:
            Network output
        """
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients using backpropagation.
        
        Args:
            X: Input data
            y: True labels
            y_pred: Model predictions
            
        Returns:
            Dictionary of gradients for each parameter
        """
        batch_size = X.shape[0]
        
        # Initial gradient from loss function
        grad = self.loss_derivative(y, y_pred)
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return grad
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            batch_size: Optional[int] = None, verbose: bool = True,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_patience: int = None,
            early_stopping_min_delta: float = 1e-4,
            num_verbose_prints: int = 10):
        """Train the network.
        
        Args:
            X: Training data
            y: Target values
            epochs: Number of training epochs
            batch_size: Size of mini-batches (None for full batch)
            verbose: Whether to print progress
            validation_data: Optional tuple of (X_val, y_val) for validation
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped
            early_stopping_min_delta: Minimum change in loss to qualify as an improvement
            num_verbose_prints: Number of times to print progress during training
            
        Returns:
            Dictionary containing training history
        """
        if self.layers[0].weights is None:
            self.initialize(X.shape[1])
            
        history = {'loss': []}
        if validation_data is not None:
            history['val_loss'] = []
            
        n_samples = X.shape[0]
        batch_size = n_samples if batch_size is None else batch_size
        
        # Early stopping setup
        if early_stopping_patience is not None:
            best_loss = float('inf')
            patience_counter = 0
            
        # Pre-compute batch indices for faster iteration
        n_batches = (n_samples + batch_size - 1) // batch_size
        batch_indices = [(i * batch_size, min((i + 1) * batch_size, n_samples)) 
                        for i in range(n_batches)]
        
        # Shuffle indices for random batch selection
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            # Shuffle data at each epoch
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # Mini-batch training with vectorized operations
            epoch_loss = 0
            for start_idx, end_idx in batch_indices:
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(batch_X)
                
                # Backward pass
                grad = self.backward(batch_X, batch_y, predictions)
                
                # Update weights
                self.optimizer.step(self.layers, batch_X, batch_y, predictions)
                
                # Accumulate loss
                epoch_loss += self.loss_fn(batch_y, predictions) * (end_idx - start_idx)
            
            # Compute average epoch loss
            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)
            
            # Compute validation loss if provided
            if validation_data is not None:
                val_pred = self.forward(validation_data[0])
                val_loss = self.loss_fn(validation_data[1], val_pred)
                history['val_loss'].append(val_loss)
                
                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < best_loss - early_stopping_min_delta:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping triggered after {epoch + 1} epochs")
                            break
            
            # Dynamic verbose printing
            print_interval = max(1, epochs // num_verbose_prints)
            if verbose and ((epoch + 1) % print_interval == 0 or epoch == 0 or epoch == epochs - 1):
                log_msg = f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}"
                if validation_data is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                print(log_msg)
                
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.forward(X)
