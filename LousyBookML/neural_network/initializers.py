"""
LousyBookML - A Machine Learning Library by LousyBook01
www.youtube.com/@LousyBook01

Made with ❤️ by LousyBook01

The Neural Network Weight Initializers Module
This module provides weight initialization methods for neural networks:
- He Normal/Uniform (for ReLU networks)
- Xavier/Glorot Normal/Uniform (for tanh/sigmoid networks)
- Random Normal/Uniform

Example:
    >>> from LousyBookML.neural_network.initializers import he_normal
    >>> # Initialize weights for a layer with 784 inputs and 128 neurons
    >>> weights = he_normal((784, 128))  # Good for ReLU activation
    >>> 
    >>> from LousyBookML.neural_network.initializers import xavier_uniform
    >>> # Initialize weights for a layer with tanh activation
    >>> weights = xavier_uniform((128, 64))  # Good for tanh/sigmoid
"""

import numpy as np
from typing import Tuple

def he_normal(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """He normal initialization.
    
    Initialize weights according to the method described in
    "Delving Deep into Rectifiers" by He et al.
    
    Args:
        shape: Shape of the weights matrix to initialize
        seed: Random seed for reproducibility
        
    Returns:
        Initialized weights matrix
    """
    if seed is not None:
        np.random.seed(seed)
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, shape)

def he_uniform(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """He uniform initialization.
    
    Initialize weights according to the method described in
    "Delving Deep into Rectifiers" by He et al.
    
    Args:
        shape: Shape of the weights matrix to initialize
        seed: Random seed for reproducibility
        
    Returns:
        Initialized weights matrix
    """
    if seed is not None:
        np.random.seed(seed)
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, shape)

def xavier_normal(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """Xavier/Glorot normal initialization.
    
    Initialize weights according to the method described in
    "Understanding the difficulty of training deep feedforward neural networks" by Glorot et al.
    
    Args:
        shape: Shape of the weights matrix to initialize
        seed: Random seed for reproducibility
        
    Returns:
        Initialized weights matrix
    """
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, shape)

def xavier_uniform(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """Xavier/Glorot uniform initialization.
    
    Initialize weights according to the method described in
    "Understanding the difficulty of training deep feedforward neural networks" by Glorot et al.
    
    Args:
        shape: Shape of the weights matrix to initialize
        seed: Random seed for reproducibility
        
    Returns:
        Initialized weights matrix
    """
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def random_normal(shape: Tuple[int, int], seed: int = None, std: float = 0.05) -> np.ndarray:
    """Random normal initialization.
    
    Args:
        shape: Shape of the weights matrix to initialize
        seed: Random seed for reproducibility
        std: Standard deviation of the normal distribution
        
    Returns:
        Initialized weights matrix
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, std, shape)

def random_uniform(shape: Tuple[int, int], seed: int = None, scale: float = 0.05) -> np.ndarray:
    """Random uniform initialization.
    
    Args:
        shape: Shape of the weights matrix to initialize
        seed: Random seed for reproducibility
        scale: Scale of the uniform distribution
        
    Returns:
        Initialized weights matrix
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-scale, scale, shape)

# Dictionary mapping initialization names to functions
INITIALIZERS = {
    'he_normal': he_normal,
    'he_uniform': he_uniform,
    'xavier_normal': xavier_normal,
    'xavier_uniform': xavier_uniform,
    'random_normal': random_normal,
    'random_uniform': random_uniform
}
