"""Test cases for the Neural Network module."""

import numpy as np
import pytest
from LousyBookML import NeuralNetwork
from LousyBookML.neural_network.activations import relu, sigmoid, tanh, softmax, leaky_relu
from LousyBookML.neural_network.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
from LousyBookML.neural_network.model import Layer
from LousyBookML.neural_network.utils import normalize_data, to_categorical

def test_activation_functions():
    """Test activation functions with various input types and edge cases."""
    x = np.array([-1.0, 0.0, 1.0])
    large_x = np.array([1000., -1000.])
    zero_x = np.zeros((5,))
    
    # Test ReLU
    assert np.array_equal(relu(x), np.array([0., 0., 1.]))
    assert np.array_equal(relu(zero_x), zero_x)
    assert np.array_equal(relu(large_x), np.array([1000., 0.]))
    
    # Test Sigmoid
    sigmoid_out = sigmoid(x)
    assert sigmoid_out.shape == x.shape
    assert np.all((sigmoid_out >= 0) & (sigmoid_out <= 1))
    # Test sigmoid with large values (should approach 0 or 1)
    assert np.allclose(sigmoid(large_x), np.array([1., 0.]), atol=1e-10)
    
    # Test Tanh
    tanh_out = tanh(x)
    assert tanh_out.shape == x.shape
    assert np.all((tanh_out >= -1) & (tanh_out <= 1))
    # Test tanh with large values (should approach -1 or 1)
    assert np.allclose(tanh(large_x), np.array([1., -1.]), atol=1e-10)
    
    # Test Softmax
    x_2d = np.array([[1, 2], [3, 4]])
    softmax_out = softmax(x_2d)
    assert np.allclose(np.sum(softmax_out, axis=1), [1., 1.])
    # Test softmax numerical stability with large numbers
    large_x_2d = np.array([[1000., 1000.], [-1000., -1000.]])
    large_softmax = softmax(large_x_2d)
    assert np.allclose(np.sum(large_softmax, axis=1), [1., 1.])

def test_losses():
    """Test loss functions."""
    y_true = np.array([[0], [1]])
    y_pred = np.array([[0.1], [0.9]])
    
    # Test all losses return valid values
    assert isinstance(mean_squared_error(y_true, y_pred), float)
    assert isinstance(binary_crossentropy(y_true, y_pred), float)
    
    y_true_cat = np.array([[1,0], [0,1]])
    y_pred_cat = np.array([[0.9,0.1], [0.1,0.9]])
    assert isinstance(categorical_crossentropy(y_true_cat, y_pred_cat), float)

def test_utils():
    """Test utility functions."""
    # Test normalize_data
    X = np.array([[1, 2], [3, 4]])
    X_norm, _, _ = normalize_data(X)
    assert X_norm.shape == X.shape
    
    # Test one_hot_encode
    y = np.array([0, 1, 2])
    y_onehot = to_categorical(y, num_classes=3)
    assert y_onehot.shape == (3, 3)
    assert np.array_equal(np.sum(y_onehot, axis=1), np.ones(3))

def test_neural_network():
    """Test neural network for both regression and classification."""
    # Simple regression test
    X = np.array([[1], [2], [3]])
    y = np.array([[2], [4], [6]])  # y = 2x
    
    model = NeuralNetwork([
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'linear'}
    ], loss='binary_crossentropy')
    
    history = model.fit(X, y, epochs=10, batch_size=1, verbose=False)
    assert isinstance(history, dict)
    assert 'loss' in history
    
    pred = model.predict(X)
    assert pred.shape == y.shape
    
    # Simple classification test
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR function
    
    model = NeuralNetwork([
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='mse')
    
    history = model.fit(X, y, epochs=10, batch_size=1, verbose=False)
    assert isinstance(history, dict)
    assert 'loss' in history
    
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.all((pred >= 0) & (pred <= 1))

def test_model_convergence():
    """Test if model can converge on simple problems."""
    # Test AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])  # AND gate
    
    model = NeuralNetwork([
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='binary_crossentropy')
    
    history = model.fit(X, y, epochs=1000, batch_size=1, verbose=False)
    
    # Check if loss decreases over time
    first_loss = np.mean(history['loss'][:10])
    last_loss = np.mean(history['loss'][-10:])
    assert last_loss < first_loss
    
    # Test OR gate
    y_or = np.array([[0], [1], [1], [1]])  # OR gate
    
    model = NeuralNetwork([
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='binary_crossentropy')
    
    history = model.fit(X, y_or, epochs=1000, batch_size=1, verbose=False)
    
    # Check if loss decreases over time
    first_loss = np.mean(history['loss'][:10])
    last_loss = np.mean(history['loss'][-10:])
    assert last_loss < first_loss
    
    # Test XOR gate
    y_xor = np.array([[0], [1], [1], [0]])  # XOR gate
    
    model = NeuralNetwork([
        {'units': 16, 'activation': 'relu'},  # Increased hidden units
        {'units': 8, 'activation': 'relu'},   # Increased hidden units
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='mse', optimizer='adam', learning_rate=0.01)
    
    history = model.fit(X, y_xor, epochs=1000, batch_size=4, verbose=False)  # More epochs, full batch
    
    # Check if loss decreases over time
    first_loss = np.mean(history['loss'][:10])
    last_loss = np.mean(history['loss'][-10:])
    assert last_loss < first_loss
    
    # Verify XOR predictions
    pred = model.predict(X)
    binary_pred = (pred > 0.5).astype(int)
    assert np.array_equal(binary_pred, y_xor), f"XOR predictions do not match. Expected {y_xor.flatten()}, got {binary_pred.flatten()}"

def test_batch_sizes():
    """Test model training with different batch sizes."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR function
    
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        model = NeuralNetwork([
            {'units': 4, 'activation': 'relu'},
            {'units': 1, 'activation': 'sigmoid'}
        ], loss='binary_crossentropy')
        
        history = model.fit(X, y, epochs=10, batch_size=batch_size, verbose=False)
        assert isinstance(history, dict)
        assert 'loss' in history
        assert len(history['loss']) == 10
        
        pred = model.predict(X)
        assert pred.shape == y.shape

def test_model_robustness():
    """Test model robustness with different input scales."""
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # Scale features
    X_scaled = X * 0.01  # Scale down instead of up for better numerical stability
    
    model = NeuralNetwork([
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='binary_crossentropy')
    
    # Train on scaled data
    history = model.fit(X_scaled, y, epochs=100, batch_size=16, verbose=False)
    
    # Loss should eventually decrease
    first_loss = np.mean(history['loss'][:10])
    last_loss = np.mean(history['loss'][-10:])
    assert last_loss < first_loss
    
    # Predictions should have correct shape and range
    pred = model.predict(X_scaled)
    assert pred.shape == y.shape
    assert np.all((pred >= 0) & (pred <= 1))

def test_layer_configuration():
    """Test different ways to configure network layers."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR function
    
    # Test with Layer objects
    model1 = NeuralNetwork([
        Layer(units=8, activation='relu'),
        Layer(units=4, activation='relu'),
        Layer(units=1, activation='sigmoid')
    ], loss='binary_crossentropy')
    
    history1 = model1.fit(X, y, epochs=1000, batch_size=1, verbose=False)
    assert history1['loss'][-1] < history1['loss'][0]
    
    # Test with traditional dict configuration
    model3 = NeuralNetwork([
        {'units': 8, 'activation': 'relu'},
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='binary_crossentropy')
    
    history3 = model3.fit(X, y, epochs=1000, batch_size=1, verbose=False)
    assert history3['loss'][-1] < history3['loss'][0]
    
    # Verify all models produce valid predictions
    for model in [model1, model3]:
        pred = model.predict(X)
        assert pred.shape == y.shape
        assert np.all((pred >= 0) & (pred <= 1))

def test_optimizers():
    """Test different optimizers."""
    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    optimizers_config = [
        {'name': 'sgd', 'params': {'momentum': 0.0, 'learning_rate': 0.1}},
        {'name': 'sgd', 'params': {'momentum': 0.9, 'learning_rate': 0.1}},  # SGD with momentum
        {'name': 'adam', 'params': {'beta1': 0.9, 'beta2': 0.999, 'learning_rate': 0.01}},
        {'name': 'rmsprop', 'params': {'decay_rate': 0.9, 'learning_rate': 0.01}}
    ]
    histories = {}

    for opt_config in optimizers_config:
        # Test each optimizer
        model = NeuralNetwork([
            {'units': 16, 'activation': 'relu'},  # Increased units for better convergence
            {'units': 8, 'activation': 'relu'},
            {'units': 1, 'activation': 'sigmoid'}
        ], optimizer=opt_config['name'], loss='binary_crossentropy', **opt_config['params'])

        history = model.fit(X, y, epochs=2000, batch_size=4, verbose=False)  # Increased epochs, full batch
        histories[f"{opt_config['name']}_{opt_config['params']}"] = history

        # Verify that loss decreases
        first_loss = np.mean(history['loss'][:10])
        last_loss = np.mean(history['loss'][-10:])
        assert last_loss < first_loss, f"Loss did not decrease for {opt_config['name']} with params {opt_config['params']}"

        # Make predictions
        pred = model.predict(X)
        binary_pred = (pred > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(binary_pred == y)
        assert accuracy > 0.75, f"Low accuracy ({accuracy}) for {opt_config['name']} with params {opt_config['params']}"

    # Verify that SGD with momentum converges faster than without momentum
    sgd_no_momentum_loss = histories[f"sgd_{{'momentum': 0.0, 'learning_rate': 0.1}}"]['loss']
    sgd_with_momentum_loss = histories[f"sgd_{{'momentum': 0.9, 'learning_rate': 0.1}}"]['loss']
    
    # Compare convergence speed (loss after 200 epochs)
    early_loss_no_momentum = np.mean(sgd_no_momentum_loss[190:210])
    early_loss_with_momentum = np.mean(sgd_with_momentum_loss[190:210])
    assert early_loss_with_momentum < early_loss_no_momentum, "SGD with momentum should converge faster"

def test_leaky_relu():
    """Test leaky ReLU activation."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    model = NeuralNetwork([
        {'units': 4, 'activation': 'leaky_relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], optimizer='adam', loss='binary_crossentropy')
    
    history = model.fit(X, y, epochs=1000, verbose=False)
    
    # Check that loss decreased
    assert history['loss'][-1] < history['loss'][0]
    
    # Test leaky ReLU function directly
    x = np.array([-2, -1, 0, 1, 2])
    output = leaky_relu(x)
    assert output[0] == -0.02  # alpha * -2
    assert output[2] == 0
    assert output[4] == 2

def test_initializers():
    """Test different weight initialization methods."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR function
    
    initializers = ['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform']
    
    for init in initializers:
        # Test with explicit Layer objects
        model = NeuralNetwork([
            Layer(units=16, activation='relu', kernel_initializer=init, seed=42),
            Layer(units=8, activation='relu', kernel_initializer=init, seed=42),
            Layer(units=1, activation='sigmoid', kernel_initializer=init, seed=42)
        ], loss='mse', optimizer='adam', learning_rate=0.01)
        
        history = model.fit(X, y, epochs=1000, batch_size=4, verbose=False)
        
        # Check if loss decreases over time
        assert history['loss'][-1] < history['loss'][0]
        
        # Test predictions
        predictions = model.predict(X)
        binary_pred = (predictions > 0.5).astype(int)
        assert np.array_equal(binary_pred, y), f"Predictions with {init} initializer do not match expected values"
        
        # Test with dictionary configuration
        model = NeuralNetwork([
            {'units': 16, 'activation': 'relu', 'kernel_initializer': init},
            {'units': 8, 'activation': 'relu', 'kernel_initializer': init},
            {'units': 1, 'activation': 'sigmoid', 'kernel_initializer': init}
        ], loss='mse', optimizer='adam', learning_rate=0.01)
        
        history = model.fit(X, y, epochs=1000, batch_size=4, verbose=False)
        assert history['loss'][-1] < history['loss'][0]
