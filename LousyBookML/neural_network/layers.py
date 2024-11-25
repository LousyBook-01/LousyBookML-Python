"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Layer Configuration Module
This module provides functions and utilities for configuring neural network layers:
- Layer configuration
- Repeated layer generation
- Layer stack management

Example:
    >>> from LousyBookML.neural_network.layers import Layer, RepeatedLayer, LayerStack
    >>> layers = LayerStack([
    ...     Layer(units=128, activation='relu'),
    ...     RepeatedLayer(count=2, units=64, activation='relu'),
    ...     Layer(units=1, activation='sigmoid')
    ... ])
    >>> model = NeuralNetwork(layers)
"""

import numpy as np
from typing import List, Union, Optional, Dict
from dataclasses import dataclass

@dataclass
class Layer:
    """Single neural network layer configuration.
    
    Args:
        units: Number of neurons in the layer
        activation: Activation function name ('relu', 'sigmoid', etc.)
        name: Optional name for the layer
        
    Example:
        >>> hidden = Layer(units=64, activation='relu', name='hidden1')
        >>> output = Layer(units=1, activation='sigmoid', name='output')
    """
    units: int
    activation: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Convert layer configuration to dictionary format.
        
        Returns:
            Dict containing units and activation function.
        """
        return {
            'units': self.units,
            'activation': self.activation
        }

@dataclass
class RepeatedLayer:
    """Configuration for multiple identical layers.
    
    Args:
        count: Number of layers to create
        units: Number of neurons in each layer
        activation: Activation function for all layers
        name_prefix: Optional prefix for layer names
        
    Example:
        >>> hidden = RepeatedLayer(
        ...     count=3,
        ...     units=64, 
        ...     activation='relu',
        ...     name_prefix='hidden'
        ... )
    """
    count: int
    units: int
    activation: str
    name_prefix: Optional[str] = None
    
    def to_layers(self) -> List[Layer]:
        """Convert to a list of Layer objects.
        
        Returns:
            List of Layer objects with identical configuration.
        """
        return [
            Layer(
                units=self.units,
                activation=self.activation,
                name=f"{self.name_prefix}_{i+1}" if self.name_prefix else None
            )
            for i in range(self.count)
        ]

class LayerStack:
    """Container for organizing multiple neural network layers.
    
    Args:
        layers: List of Layer and/or RepeatedLayer objects
        
    Example:
        >>> stack = LayerStack([
        ...     Layer(units=128, activation='relu'),
        ...     RepeatedLayer(count=2, units=64, activation='relu'),
        ...     Layer(units=1, activation='sigmoid')
        ... ])
        >>> architecture = stack.to_architecture()
    """
    def __init__(self, layers: List[Union[Layer, RepeatedLayer]]):
        self.layer_configs = layers
        
    def to_architecture(self) -> List[Dict[str, Union[int, str]]]:
        """Convert layer stack to architecture format.
        
        Returns:
            List of layer configuration dictionaries.
        """
        architecture = []
        
        for layer in self.layer_configs:
            if isinstance(layer, Layer):
                architecture.append(layer.to_dict())
            elif isinstance(layer, RepeatedLayer):
                architecture.extend(l.to_dict() for l in layer.to_layers())
                
        return architecture
    
    def __len__(self) -> int:
        """Get total number of layers in the stack.
        
        Returns:
            Total number of layers after expanding RepeatedLayers.
        """
        return sum(
            1 if isinstance(layer, Layer) else layer.count
            for layer in self.layer_configs
        )
