"""Optimizers for neural network training."""

import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def initialize(self, params):
        """Initialize velocity for each parameter."""
        self.velocities = {key: np.zeros_like(value) for key, value in params.items()}
    
    def update(self, params, grads):
        """Update parameters using SGD with momentum."""
        if not self.velocities:
            self.initialize(params)
        
        for key in params:
            grad_key = 'd' + key  # Convert 'W1' to 'dW1'
            self.velocities[key] = (self.momentum * self.velocities[key] - 
                                  self.learning_rate * grads[grad_key])
            params[key] += self.velocities[key]
        return params

class Adam:
    """Adam optimizer."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
    
    def initialize(self, params):
        """Initialize moment estimates."""
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}
    
    def update(self, params, grads):
        """Update parameters using Adam optimization."""
        if not self.m or not self.v:
            self.initialize(params)
        
        self.t += 1
        
        for key in params:
            grad_key = 'd' + key  # Convert 'W1' to 'dW1'
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[grad_key]
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[grad_key])
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[key] -= (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)
        
        return params

class RMSprop:
    """RMSprop optimizer."""
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def initialize(self, params):
        """Initialize cache for squared gradients."""
        self.cache = {key: np.zeros_like(value) for key, value in params.items()}
    
    def update(self, params, grads):
        """Update parameters using RMSprop."""
        if not self.cache:
            self.initialize(params)
        
        for key in params:
            grad_key = 'd' + key  # Convert 'W1' to 'dW1'
            self.cache[key] = (self.decay_rate * self.cache[key] + 
                             (1 - self.decay_rate) * np.square(grads[grad_key]))
            params[key] -= (self.learning_rate * grads[grad_key] / 
                          (np.sqrt(self.cache[key]) + self.epsilon))
        
        return params
