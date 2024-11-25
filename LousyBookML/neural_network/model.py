"""Neural network model implementation."""

import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
from .activations import ACTIVATION_FUNCTIONS
from .losses import LOSS_FUNCTIONS
from .optimizers import OPTIMIZERS
from .initializers import INITIALIZERS

class Layer:
    """Neural network layer."""
    
    def __init__(self, units: int, activation: str = 'linear', 
                 kernel_initializer: str = 'xavier_uniform', seed: Optional[int] = None,
                 batch_norm: bool = False):
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.seed = seed
        self.batch_norm = batch_norm
        
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation function: {activation}")
        if kernel_initializer not in INITIALIZERS:
            raise ValueError(f"Unknown initializer: {kernel_initializer}")
            
        self.activation_fn = ACTIVATION_FUNCTIONS[activation]['forward']
        self.activation_derivative = ACTIVATION_FUNCTIONS[activation]['backward']
        self.initializer = INITIALIZERS[kernel_initializer]
        
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None
        
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
                normalized = (output - batch_mean) / np.sqrt(batch_var)
            else:
                # Use running statistics for inference
                normalized = ((output - self.running_mean) / 
                            np.sqrt(self.running_var + self.epsilon))
            
            # Scale and shift
            output = self.gamma * normalized + self.beta
        
        self.output = self.activation_fn(output)
        return self.output

class NeuralNetwork:
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
        self.loss_fn = LOSS_FUNCTIONS[loss]
        
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            batch_size: Optional[int] = None, verbose: bool = True,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_patience: int = None,
            early_stopping_min_delta: float = 1e-4):
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
            
            if verbose and (epoch + 1) % 100 == 0:
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
