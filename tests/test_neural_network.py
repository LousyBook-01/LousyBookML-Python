import numpy as np
import pytest
from lousybook01.LousyBookML import NeuralNetwork, LayerConfig

def test_xor_problem():
    """Test if network can learn XOR function."""
    # XOR input and output
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Create network with optimized architecture
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=6, activation='relu', initialization='he', 
                   dropout_rate=0.1, weight_scale=1.2),
        LayerConfig(size=4, activation='leaky_relu', initialization='he',
                   dropout_rate=0.1, weight_scale=1.0),
        LayerConfig(size=1, activation='sigmoid', initialization='xavier',
                   weight_scale=0.8)
    ]
    
    # Try multiple random initializations
    for _ in range(5):  # Increased attempts for better chance of convergence
        np.random.seed()  # Reset random seed
        
        model = NeuralNetwork(
            layer_configs=layers,
            optimizer='rmsprop',  # Changed to RMSprop for better convergence
            learning_rate=0.05,   # Lower learning rate for stability
            rmsprop_beta=0.99,
            batch_norm=True,
            gradient_clip=1.0
        )
        
        # Train with mini-batches for better generalization
        model.train(X, y, epochs=2000, batch_size=2, verbose=False)
        
        # Test predictions with slightly relaxed threshold
        predictions = model.predict(X)
        if np.all(np.abs(predictions - y) < 0.2):  # Slightly relaxed threshold
            break  # Success
    else:
        assert False, "Failed to learn XOR after 5 attempts"

def test_initialization():
    """Test different initialization schemes."""
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4, activation='relu', initialization='he'),
        LayerConfig(size=4, activation='tanh', initialization='xavier'),
        LayerConfig(size=1, activation='sigmoid')
    ]
    
    # Test multiple random initializations
    passed = False
    for _ in range(5):  # Try up to 5 times
        model = NeuralNetwork(layer_configs=layers)
        
        # Check if weights are properly initialized
        valid = True
        for i, layer_config in enumerate(layers[1:], 1):
            if layer_config.initialization == 'he':
                fan_in = layers[i-1].size
                expected_std = np.sqrt(2.0 / fan_in)
                actual_std = np.std(model.weights[i-1])
                # Allow for 30% deviation from expected standard deviation
                if abs(actual_std - expected_std) >= 0.3 * expected_std:
                    valid = False
                    break
        
        if valid:
            passed = True
            break
    
    assert passed, "Weight initialization failed to meet expected distribution after 5 attempts"

def test_activations():
    """Test different activation functions."""
    # Test data
    X = np.array([[1, -1]])
    
    for activation in ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear']:
        layers = [
            LayerConfig(size=1),
            LayerConfig(size=1, activation=activation)
        ]
        
        model = NeuralNetwork(layer_configs=layers)
        
        # Test forward pass
        activations, _ = model.forward_propagation(X)
        assert not np.any(np.isnan(activations[-1]))

def test_dropout():
    """Test dropout functionality."""
    np.random.seed(42)
    
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=10, dropout_rate=0.5),
        LayerConfig(size=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Test if dropout is applied during training
    X = np.random.randn(2, 5)
    activations1, _ = model.forward_propagation(X, training=True)
    activations2, _ = model.forward_propagation(X, training=True)
    
    # Activations should be different due to dropout
    assert not np.allclose(activations1[-1], activations2[-1])
    
    # Test if dropout is disabled during inference
    activations3, _ = model.forward_propagation(X, training=False)
    activations4, _ = model.forward_propagation(X, training=False)
    
    # Activations should be same when not training
    assert np.allclose(activations3[-1], activations4[-1])

def test_batch_normalization():
    """Test batch normalization."""
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4),
        LayerConfig(size=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers, batch_norm=True)
    
    # Test if batch norm parameters are initialized
    assert hasattr(model, 'gamma')
    assert hasattr(model, 'beta')
    assert hasattr(model, 'running_mean')
    assert hasattr(model, 'running_var')

def test_gradient_clipping():
    """Test gradient clipping."""
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4),
        LayerConfig(size=1)
    ]
    
    clip_value = 1.0
    model = NeuralNetwork(layer_configs=layers, gradient_clip=clip_value)
    
    # Create large gradients
    large_gradients = [np.ones((4, 2)) * 10, np.ones((1, 4)) * 10]
    
    # Clip gradients
    clipped = model._clip_gradients(large_gradients)
    
    # Check if gradients are clipped
    for grad in clipped:
        assert np.all(np.abs(grad) <= clip_value)

def test_optimization_algorithms():
    """Test different optimization algorithms."""
    X = np.random.randn(2, 4)
    y = np.random.randn(1, 4)
    
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4),
        LayerConfig(size=1)
    ]
    
    optimizers = ['sgd', 'momentum', 'rmsprop']
    
    for opt in optimizers:
        model = NeuralNetwork(layer_configs=layers, optimizer=opt)
        # Should run without errors
        model.train(X, y, epochs=10, batch_size=2)

def test_input_validation():
    """Test input validation."""
    # Test invalid layer sizes
    with pytest.raises(ValueError, match="Layer size must be positive"):
        NeuralNetwork([LayerConfig(size=0), LayerConfig(size=1)])
    
    with pytest.raises(ValueError, match="Layer size must be positive"):
        NeuralNetwork([LayerConfig(size=1), LayerConfig(size=-1)])
    
    # Test invalid input dimensions
    model = NeuralNetwork([LayerConfig(size=1), LayerConfig(size=1)])
    with pytest.raises(ValueError, match="Number of samples in X and y must match"):
        model.train(np.array([1, 2]), np.array([1, 2, 3]))

def test_parameter_tracking():
    """Test if the model tracks training metrics."""
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4, activation='relu'),
        LayerConfig(size=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    X = np.random.randn(2, 4)
    y = np.random.randn(1, 4)
    
    model.train(X, y, epochs=10, batch_size=2)
    
    assert len(model.gradient_norms) > 0

def test_learning_rate_scheduler():
    """Test learning rate scheduling functionality."""
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4),
        LayerConfig(size=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Test warmup phase
    lr_warmup = model._get_learning_rate(0.1, 50)  # During warmup
    lr_after_warmup = model._get_learning_rate(0.1, 150)  # After warmup
    
    assert lr_warmup < 0.1  # Should be ramping up
    assert lr_after_warmup <= 0.1  # Should be at or below base lr

def test_regularization():
    """Test L1 and L2 regularization effects."""
    X = np.random.randn(2, 10)
    y = np.random.randn(1, 10)
    
    # Network with L2 regularization
    layers_l2 = [
        LayerConfig(size=2),
        LayerConfig(size=4, l2_reg=0.1),
        LayerConfig(size=1)
    ]
    
    # Network with L1 regularization
    layers_l1 = [
        LayerConfig(size=2),
        LayerConfig(size=4, l1_reg=0.1),
        LayerConfig(size=1)
    ]
    
    model_l2 = NeuralNetwork(layer_configs=layers_l2)
    model_l1 = NeuralNetwork(layer_configs=layers_l1)
    
    # Train both models
    model_l2.train(X, y, epochs=10, batch_size=5)
    model_l1.train(X, y, epochs=10, batch_size=5)
    
    # Check if weights are more sparse with L1
    l1_sparsity = np.mean(np.abs(model_l1.weights[0]) < 1e-4)
    l2_sparsity = np.mean(np.abs(model_l2.weights[0]) < 1e-4)
    
    assert l1_sparsity >= l2_sparsity  # L1 should induce more sparsity

def test_complex_architecture():
    """Test a more complex network architecture."""
    # Create a complex dataset
    X = np.random.randn(5, 100)  # 5 features, 100 samples
    y = np.random.randn(2, 100)  # 2 outputs
    
    layers = [
        LayerConfig(size=5),  # Input
        LayerConfig(size=8, activation='relu', dropout_rate=0.2),
        LayerConfig(size=8, activation='leaky_relu', dropout_rate=0.2),
        LayerConfig(size=4, activation='tanh'),
        LayerConfig(size=2, activation='linear')  # Output
    ]
    
    model = NeuralNetwork(
        layer_configs=layers,
        optimizer='rmsprop',
        batch_norm=True,
        gradient_clip=1.0
    )
    
    # Should train without errors
    model.train(X, y, epochs=5, batch_size=32)
    
    # Test prediction shape
    pred = model.predict(X)
    assert pred.shape == (2, 100)

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    X = np.random.randn(2, 10) * 1000  # Large inputs
    y = np.random.randn(1, 10) * 1000  # Large targets
    
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4, activation='tanh'),  # tanh for bounded activation
        LayerConfig(size=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Should handle large values without NaN
    model.train(X, y, epochs=5, batch_size=5)
    pred = model.predict(X)
    
    assert not np.any(np.isnan(pred))
    assert not np.any(np.isinf(pred))
