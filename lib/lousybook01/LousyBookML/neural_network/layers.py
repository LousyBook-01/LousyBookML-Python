"""
Layer configurations and implementations for neural networks.
"""

class LayerConfig:
    """Configuration for a neural network layer."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = 'linear',
                 initialization: str = 'he',
                 dropout_rate: float = 0.0,
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.0,
                 batch_norm: bool = False):
        """Initialize layer configuration.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function ('linear', 'relu', 'sigmoid', etc.)
            initialization: Weight initialization method ('he', 'xavier', etc.)
            dropout_rate: Dropout rate (0.0 means no dropout)
            l1_reg: L1 regularization coefficient
            l2_reg: L2 regularization coefficient
            batch_norm: Whether to use batch normalization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initialization = initialization
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_norm = batch_norm
        
        # Add size attribute for compatibility
        self.size = output_dim
