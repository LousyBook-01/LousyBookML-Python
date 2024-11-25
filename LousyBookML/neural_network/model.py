"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Model Module
This module provides the core neural network implementation:
- Feed-forward Neural Networks
- Customizable Layer Architecture
- Multiple Activation Functions
- Mini-batch Training Support
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from .activations import ACTIVATION_FUNCTIONS
from .losses import LOSS_FUNCTIONS
from .layers import Layer, RepeatedLayer, LayerStack
from .optimizers import SGD, Adam, RMSprop

class NeuralNetwork:
    """A flexible implementation of a feed-forward neural network.
    
    Attributes:
        architecture (List[Dict]): List of layer configurations
        learning_rate (float): Learning rate for gradient descent
        parameters (Dict): Weight and bias matrices for each layer
        cache (Dict): Cache for forward and backward passes
        _initialized (bool): Whether the network is initialized
        loss_name (str): Loss function name
        loss_function (callable): Loss function
        optimizer: Optimizer instance
        
    Example:
        >>> # Using the new layer configuration system
        >>> model = NeuralNetwork(
        ...     LayerStack([
        ...         Layer(units=64, activation='relu'),
        ...         RepeatedLayer(count=2, units=32, activation='relu'),
        ...         Layer(units=1, activation='sigmoid')
        ...     ]),
        ...     loss='binary_crossentropy',
        ...     optimizer='adam',
        ...     learning_rate=0.001
        ... )
        >>> model.fit(X_train, y_train, epochs=100)
        
        >>> # Using the traditional dictionary-based configuration
        >>> model = NeuralNetwork([
        ...     {'units': 64, 'activation': 'relu'},
        ...     {'units': 1, 'activation': 'sigmoid'}
        ... ])
    """
    
    def __init__(self, 
                 architecture: Union[LayerStack, List[Dict[str, Union[int, str]]], List[Layer]], 
                 loss: str = 'mean_squared_error',
                 optimizer: str = 'sgd',
                 learning_rate: float = 0.01,
                 **optimizer_params):
        """Initialize the neural network.
        
        Args:
            architecture: Network architecture specification. Can be:
                - LayerStack object
                - List of Layer objects
                - List of layer configuration dictionaries
            loss: Loss function name
            optimizer: Optimizer name ('sgd', 'adam', or 'rmsprop')
            learning_rate: Learning rate for the optimizer
            **optimizer_params: Additional optimizer parameters (e.g., momentum, beta1, beta2)
        """
        self.architecture = self._process_architecture(architecture)
        self.loss_name = loss
        self.loss_function = LOSS_FUNCTIONS[loss]
        
        # Initialize optimizer
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate=learning_rate, 
                               momentum=optimizer_params.get('momentum', 0.0))
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate=learning_rate,
                                beta1=optimizer_params.get('beta1', 0.9),
                                beta2=optimizer_params.get('beta2', 0.999),
                                epsilon=optimizer_params.get('epsilon', 1e-8))
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=learning_rate,
                                   decay_rate=optimizer_params.get('decay_rate', 0.9),
                                   epsilon=optimizer_params.get('epsilon', 1e-8))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        self.parameters = {}
        self.cache = {}
        self._initialized = False

    def _process_architecture(self, architecture):
        """Convert architecture to standard format."""
        if isinstance(architecture, LayerStack):
            return architecture.to_architecture()
        elif isinstance(architecture, list):
            if all(isinstance(layer, Layer) for layer in architecture):
                return [layer.to_dict() for layer in architecture]
            return architecture
        raise ValueError("Architecture must be LayerStack, list of Layers, or list of dicts")

    def initialize_parameters(self, input_dim: int) -> None:
        """Initialize network parameters.
        
        Args:
            input_dim: Dimension of input features
        """
        prev_units = input_dim
        
        for i, layer in enumerate(self.architecture):
            units = layer['units']
            # He initialization
            self.parameters[f'W{i+1}'] = np.random.randn(prev_units, units) * np.sqrt(2. / prev_units)
            self.parameters[f'b{i+1}'] = np.zeros((1, units))
            prev_units = units
        
        self._initialized = True

    def fit(self, 
           X: np.ndarray, 
           y: np.ndarray, 
           epochs: int = 100,
           batch_size: Optional[int] = None,
           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
           verbose: bool = True) -> Dict[str, List[float]]:
        """Train the neural network.
        
        Args:
            X: Input features
            y: Target values
            epochs: Number of training epochs
            batch_size: Size of each training batch
            validation_data: Tuple of (X_val, y_val) for validation
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        if not self._initialized:
            self.initialize_parameters(X.shape[1])
        
        history = {'loss': [], 'val_loss': []}
        n_samples = X.shape[0]
        batch_size = batch_size or n_samples
        
        for epoch in range(epochs):
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # Forward pass
                self._forward(batch_X)
                
                # Backward pass
                gradients = self._backward(batch_X, batch_y)
                
                # Update parameters
                self._update_parameters(gradients)
            
            # Compute training loss
            predictions = self.predict(X)
            loss = self.loss_function(y, predictions)
            history['loss'].append(loss)
            
            # Compute validation loss if provided
            if validation_data is not None:
                val_pred = self.predict(validation_data[0])
                val_loss = self.loss_function(validation_data[1], val_pred)
                history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                log_msg = f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}"
                if validation_data is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                print(log_msg)
        
        return history

    def _forward(self, X: np.ndarray) -> None:
        """Forward pass through the network.
        
        Args:
            X: Input features
            
        Returns:
            Network output
        """
        self.cache = {}
        self.cache['A0'] = X
        
        for i, layer in enumerate(self.architecture, 1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            
            # Linear transformation
            Z = np.dot(self.cache[f'A{i-1}'], W) + b
            self.cache[f'Z{i}'] = Z
            
            # Activation
            activation_fn = ACTIVATION_FUNCTIONS[layer['activation']]
            A = activation_fn(Z)
            self.cache[f'A{i}'] = A
        
        return self.cache[f'A{len(self.architecture)}']

    def _backward(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary of gradients
        """
        m = X.shape[0]
        gradients = {}
        
        # Get the derivative of the loss with respect to the output
        if self.loss_name == 'categorical_crossentropy' and self.architecture[-1]['activation'] == 'softmax':
            dA = self.cache[f'A{len(self.architecture)}'] - y
        else:
            dZ = self.cache[f'A{len(self.architecture)}'] - y
            dA = dZ * ACTIVATION_FUNCTIONS[self.architecture[-1]['activation'] + '_derivative'](self.cache[f'Z{len(self.architecture)}'])
        
        for layer in range(len(self.architecture), 0, -1):
            current_cache = self.cache[f'A{layer-1}']
            
            # Calculate gradients
            gradients[f'dW{layer}'] = (1/m) * np.dot(current_cache.T, dA)
            gradients[f'db{layer}'] = (1/m) * np.sum(dA, axis=0, keepdims=True)
            
            if layer > 1:
                dZ = np.dot(dA, self.parameters[f'W{layer}'].T)
                activation = self.architecture[layer-2]['activation']
                dA = dZ * ACTIVATION_FUNCTIONS[activation + '_derivative'](self.cache[f'Z{layer-1}'])
        
        return gradients

    def _update_parameters(self, gradients: Dict[str, np.ndarray]) -> None:
        """Update network parameters using the optimizer.
        
        Args:
            gradients: Dictionary of gradients
        """
        # Update parameters using the optimizer
        updated_params = self.optimizer.update(self.parameters, gradients)
        
        # Update the network parameters
        self.parameters.update(updated_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if not self._initialized:
            raise ValueError("Model must be trained before making predictions")
        return self._forward(X)
