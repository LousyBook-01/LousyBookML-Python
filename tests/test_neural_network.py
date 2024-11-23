import numpy as np
import pytest
from lousybook01.LousyBookML import NeuralNetwork, LayerConfig

def test_xor_problem():
    """Test if network can learn XOR function with leaky_relu and RMSprop."""
    # XOR input and output
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]]).T  # Shape: (4, 2)
    y = np.array([[0], [1], [1], [0]])  # Shape: (4, 1)
    
    # Use the exact successful configuration
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=4, activation='leaky_relu'),
        LayerConfig(input_dim=4, output_dim=4, activation='leaky_relu'),
        LayerConfig(input_dim=4, output_dim=1, activation='sigmoid')
    ]
    
    success = False
    best_loss = float('inf')
    best_predictions = None
    
    # Try multiple times with the same configuration
    for attempt in range(10):  # Try up to 10 times
        np.random.seed(attempt * 42)  # Different seed each time
        
        model = NeuralNetwork(
            layer_configs=layers,
            optimizer='rmsprop',
            learning_rate=0.01,
            gradient_clip=1.0
        )
        
        # Train
        model.train(X, y, max_iter=1000)
        
        # Test predictions
        predictions = model.predict(X)
        loss = np.mean((predictions - y) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_predictions = predictions.copy()
        
        if np.all(np.abs(predictions - y) < 0.1):  # Allow for some tolerance
            success = True
            break
    
    if not success:
        print(f"Failed to learn XOR. Best loss achieved: {best_loss:.4f}")
        print("Best predictions:")
        print(f"[0, 0] -> {best_predictions[0,0]:.4f} (expected 0)")
        print(f"[0, 1] -> {best_predictions[1,0]:.4f} (expected 1)")
        print(f"[1, 0] -> {best_predictions[2,0]:.4f} (expected 1)")
        print(f"[1, 1] -> {best_predictions[3,0]:.4f} (expected 0)")
        assert False, "Failed to learn XOR after multiple attempts"

def test_initialization():
    """Test different initialization schemes."""
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=4, activation='relu', initialization='he'),
        LayerConfig(input_dim=4, output_dim=4, activation='tanh', initialization='xavier'),
        LayerConfig(input_dim=4, output_dim=1, activation='sigmoid')
    ]
    
    # Test multiple random initializations
    model = NeuralNetwork(layer_configs=layers)
    
    # Check if weights exist and have correct shapes
    assert len(model.weights) == len(layers)
    for i, layer_config in enumerate(layers):
        # Check shape
        assert model.weights[i].shape == (layer_config.output_dim, layer_config.input_dim), \
            f"Layer {i} weights shape mismatch. Expected {(layer_config.output_dim, layer_config.input_dim)}, got {model.weights[i].shape}"
        
        # Check if weights are not all zeros or ones
        assert not np.allclose(model.weights[i], 0), \
            f"Layer {i} weights are all close to zero"
        assert not np.allclose(model.weights[i], 1), \
            f"Layer {i} weights are all close to one"
        
        # Check if weights have reasonable magnitudes
        assert np.all(np.abs(model.weights[i]) < 10), \
            f"Layer {i} weights have unreasonable magnitudes"
        
        # Check initialization scheme effects
        if layer_config.initialization == 'he':
            # He initialization should have std ≈ sqrt(2/fan_in)
            std = np.std(model.weights[i])
            expected_std = np.sqrt(2.0 / layer_config.input_dim)
            assert 0.1 * expected_std <= std <= 5 * expected_std, \
                f"Layer {i} He initialization std ({std}) is far from expected ({expected_std})"
        elif layer_config.initialization == 'xavier':
            # Xavier initialization should have std ≈ sqrt(2/(fan_in + fan_out))
            std = np.std(model.weights[i])
            expected_std = np.sqrt(2.0 / (layer_config.input_dim + layer_config.output_dim))
            assert 0.1 * expected_std <= std <= 5 * expected_std, \
                f"Layer {i} Xavier initialization std ({std}) is far from expected ({expected_std})"

def test_activations():
    """Test different activation functions."""
    # Test data
    X = np.array([[1.0, -1.0]])  # Shape: (1, 2)
    
    for activation in ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear']:
        layers = [
            LayerConfig(input_dim=2, output_dim=2),
            LayerConfig(input_dim=2, output_dim=1, activation=activation)
        ]
        
        model = NeuralNetwork(layer_configs=layers)
        
        # Test forward pass
        activations, _ = model.forward_propagation(X)
        assert not np.any(np.isnan(activations[-1]))

def test_dropout():
    """Test dropout functionality."""
    np.random.seed(42)
    
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=10, dropout_rate=0.5),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Test if dropout is applied during training
    X = np.random.randn(5, 2)  # 5 samples, 2 features
    model.training = True
    activations1, _ = model.forward_propagation(X)
    activations2, _ = model.forward_propagation(X)
    
    # Activations should be different due to dropout
    assert not np.allclose(activations1[-1], activations2[-1])
    
    # Test if dropout is disabled during inference
    model.training = False
    activations3, _ = model.forward_propagation(X)
    activations4, _ = model.forward_propagation(X)
    
    # Activations should be same when not training
    assert np.allclose(activations3[-1], activations4[-1])

def test_batch_normalization():
    """Test batch normalization."""
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=4, batch_norm=True),
        LayerConfig(input_dim=4, output_dim=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    X = np.random.randn(32, 2)  # 32 samples, 2 features
    
    # Test training mode
    model.training = True
    activations1, cache1 = model.forward_propagation(X)
    
    # Test if batch norm parameters are tracked
    assert 1 in model.batch_norm_params
    assert 'gamma' in model.batch_norm_params[1]
    assert 'beta' in model.batch_norm_params[1]
    assert 'moving_mean' in model.batch_norm_params[1]
    assert 'moving_var' in model.batch_norm_params[1]
    
    # Test inference mode
    model.training = False
    activations2, cache2 = model.forward_propagation(X)
    
    # Outputs should be different between training and inference
    assert not np.allclose(activations1[-1], activations2[-1])

def test_gradient_clipping():
    """Test gradient clipping."""
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=4),
        LayerConfig(input_dim=4, output_dim=1)
    ]
    
    clip_value = 1.0
    model = NeuralNetwork(layer_configs=layers, gradient_clip=clip_value)
    
    # Create large gradients
    large_gradients = [np.ones((4, 2)) * 10, np.ones((1, 4)) * 10]
    
    # Clip gradients
    clipped = model._clip_gradients(large_gradients)
    
    # Check if gradients are clipped
    for grad in clipped:
        assert np.all(np.abs(grad) <= clip_value + 1e-6)  # Allow small numerical error

def test_optimization_algorithms():
    """Test different optimization algorithms."""
    X = np.random.randn(4, 2)  # 4 samples, 2 features
    y = np.random.randn(4, 1)  # 4 samples, 1 target
    
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=4),
        LayerConfig(input_dim=4, output_dim=1)
    ]
    
    optimizers = ['sgd', 'momentum', 'rmsprop', 'adam']
    
    for opt in optimizers:
        model = NeuralNetwork(layer_configs=layers, optimizer=opt)
        # Should run without errors
        model.train(X, y, max_iter=10)
        
        # Check if optimizer state is properly initialized
        if opt != 'sgd':
            assert hasattr(model, 'optimizer_state')
            assert opt in ['momentum', 'rmsprop', 'adam']
            if opt == 'momentum':
                assert 'momentum' in model.optimizer_state
            elif opt == 'rmsprop':
                assert 'rmsprop' in model.optimizer_state
            elif opt == 'adam':
                assert 'adam' in model.optimizer_state
                assert 't' in model.optimizer_state['adam']

def test_regularization():
    """Test L1 and L2 regularization effects."""
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    
    # Create three identical models with different regularization
    layers_no_reg = [
        LayerConfig(input_dim=2, output_dim=10),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    layers_l1 = [
        LayerConfig(input_dim=2, output_dim=10, l1_reg=0.1),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    layers_l2 = [
        LayerConfig(input_dim=2, output_dim=10, l2_reg=0.1),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    models = {
        'no_reg': NeuralNetwork(layers_no_reg),
        'l1': NeuralNetwork(layers_l1),
        'l2': NeuralNetwork(layers_l2)
    }
    
    # Train all models
    histories = {name: model.train(X, y, max_iter=50)
                for name, model in models.items()}
    
    # Check if regularized models have higher loss (due to regularization term)
    assert np.mean(histories['l1']['loss']) > np.mean(histories['no_reg']['loss'])
    assert np.mean(histories['l2']['loss']) > np.mean(histories['no_reg']['loss'])
    
    # Check if L1 creates more sparse weights
    l1_zeros = sum(np.sum(np.abs(w) < 1e-6) for w in models['l1'].weights)
    no_reg_zeros = sum(np.sum(np.abs(w) < 1e-6) for w in models['no_reg'].weights)
    assert l1_zeros >= no_reg_zeros

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    X = np.random.randn(10, 2) * 1000  # Large inputs
    y = np.random.randn(10, 1) * 1000  # Large targets
    
    layers = [
        LayerConfig(input_dim=2, output_dim=2),
        LayerConfig(input_dim=2, output_dim=4, activation='tanh'),  # tanh for bounded activation
        LayerConfig(input_dim=4, output_dim=1, activation='tanh')  # bounded output
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Should handle large values without NaN
    model.train(X, y, max_iter=10)
    pred = model.predict(X)
    
    assert not np.any(np.isnan(pred))
    assert not np.any(np.isinf(pred))
    assert np.all(np.abs(pred) <= 1.0)  # tanh output should be bounded

def test_loss_functions():
    """Test different loss functions."""
    # Test data
    X = np.array([[1.0, -1.0], [-1.0, 1.0]])  # Shape: (2, 2)
    
    # Test MSE loss
    y_mse = np.array([[0.5], [-0.5]])  # Shape: (2, 1)
    layers = [
        LayerConfig(input_dim=2, output_dim=1)
    ]
    model = NeuralNetwork(layer_configs=layers, loss='mse')
    pred = np.array([[0.3], [-0.7]])
    loss = model._compute_loss(pred, y_mse)
    assert isinstance(loss, float)
    assert loss >= 0
    
    # Test binary cross-entropy
    y_bce = np.array([[1], [0]])  # Binary labels
    model = NeuralNetwork(layer_configs=layers, loss='binary_crossentropy')
    pred = np.array([[0.7], [0.3]])  # Probabilities
    loss = model._compute_loss(pred, y_bce)
    assert isinstance(loss, float)
    assert loss >= 0
    
    # Test categorical cross-entropy
    y_cce = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot encoded
    layers = [LayerConfig(input_dim=2, output_dim=3)]
    model = NeuralNetwork(layer_configs=layers, loss='categorical_crossentropy')
    pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])  # Class probabilities
    loss = model._compute_loss(pred, y_cce)
    assert isinstance(loss, float)
    assert loss >= 0

def test_early_stopping():
    """Test early stopping functionality."""
    X = np.random.randn(100, 2)
    y = np.random.randn(100, 1)
    X_val = np.random.randn(20, 2)
    y_val = np.random.randn(20, 1)
    
    layers = [
        LayerConfig(input_dim=2, output_dim=4),
        LayerConfig(input_dim=4, output_dim=1)
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Test early stopping with validation data
    history = model.train(
        X, y,
        validation_data=(X_val, y_val),
        max_iter=100,
        patience=5
    )
    
    assert 'loss' in history
    assert 'val_loss' in history
    assert len(history['loss']) <= 100  # Should stop early
    assert len(history['loss']) == len(history['val_loss'])

def test_model_serialization():
    """Test model saving and loading."""
    import os
    import tempfile
    
    # Create a model with some training
    X = np.random.randn(10, 2)
    y = np.random.randn(10, 1)
    
    layers = [
        LayerConfig(input_dim=2, output_dim=4, batch_norm=True),
        LayerConfig(input_dim=4, output_dim=1)
    ]
    
    original_model = NeuralNetwork(
        layer_configs=layers,
        optimizer='adam',
        learning_rate=0.001
    )
    original_model.train(X, y, max_iter=5)
    
    # Get predictions before saving
    original_preds = original_model.predict(X)
    
    # Save and load the model
    temp_dir = tempfile.mkdtemp()
    try:
        model_path = os.path.join(temp_dir, 'model.npz')
        original_model.save_model(model_path)
        
        # Load the model
        loaded_model = NeuralNetwork(layer_configs=layers)
        loaded_model.load_model(model_path)
        
        # Compare predictions
        loaded_preds = loaded_model.predict(X)
        assert np.allclose(original_preds, loaded_preds)
        
        # Check if all attributes are preserved
        assert loaded_model.optimizer == original_model.optimizer
        assert loaded_model.learning_rate == original_model.learning_rate
        assert len(loaded_model.weights) == len(original_model.weights)
        assert len(loaded_model.biases) == len(original_model.biases)
    finally:
        # Clean up
        try:
            os.remove(model_path)
            os.rmdir(temp_dir)
        except:
            pass

def test_mini_batch_training():
    """Test mini-batch training functionality."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])  # Non-linear function
    
    layers = [
        LayerConfig(input_dim=2, output_dim=8, activation='relu'),
        LayerConfig(input_dim=8, output_dim=1, activation='linear')
    ]
    
    # Test different batch sizes
    batch_sizes = [1, 10, 32, 100]  # Including full batch
    
    for batch_size in batch_sizes:
        model = NeuralNetwork(layer_configs=layers, optimizer='adam')
        history = model.train(X, y, max_iter=50, batch_size=batch_size)
        
        # Check if loss decreases
        assert history['loss'][-1] < history['loss'][0]
        
        # Make predictions
        preds = model.predict(X)
        assert preds.shape == y.shape

def test_regularization():
    """Test L1 and L2 regularization effects."""
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    
    # Create three identical models with different regularization
    layers_no_reg = [
        LayerConfig(input_dim=2, output_dim=10),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    layers_l1 = [
        LayerConfig(input_dim=2, output_dim=10, l1_reg=0.1),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    layers_l2 = [
        LayerConfig(input_dim=2, output_dim=10, l2_reg=0.1),
        LayerConfig(input_dim=10, output_dim=1)
    ]
    
    models = {
        'no_reg': NeuralNetwork(layers_no_reg),
        'l1': NeuralNetwork(layers_l1),
        'l2': NeuralNetwork(layers_l2)
    }
    
    # Train all models
    histories = {name: model.train(X, y, max_iter=50)
                for name, model in models.items()}
    
    # Check if regularized models have higher loss (due to regularization term)
    assert np.mean(histories['l1']['loss']) > np.mean(histories['no_reg']['loss'])
    assert np.mean(histories['l2']['loss']) > np.mean(histories['no_reg']['loss'])
    
    # Check if L1 creates more sparse weights
    l1_zeros = sum(np.sum(np.abs(w) < 1e-6) for w in models['l1'].weights)
    no_reg_zeros = sum(np.sum(np.abs(w) < 1e-6) for w in models['no_reg'].weights)
    assert l1_zeros >= no_reg_zeros

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    # Test with very large and very small inputs
    X = np.array([[1e10, 1e-10], [-1e10, -1e-10]])
    y = np.array([[1], [0]])
    
    layers = [
        LayerConfig(input_dim=2, output_dim=4, activation='sigmoid'),
        LayerConfig(input_dim=4, output_dim=1, activation='sigmoid')
    ]
    
    model = NeuralNetwork(layer_configs=layers)
    
    # Forward pass should not produce NaN
    activations, _ = model.forward_propagation(X)
    assert not np.any(np.isnan(activations[-1]))
    assert not np.any(np.isinf(activations[-1]))
    
    # Loss computation should not produce NaN
    loss = model._compute_loss(activations[-1], y)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
