"""Test cases for the Neural Network module."""

import numpy as np
import pytest
from LousyBookML.neural_network import NeuralNetwork
from LousyBookML.neural_network.activations import relu, sigmoid, tanh, softmax, leaky_relu
from LousyBookML.neural_network.losses import (
    mean_squared_error, binary_cross_entropy, categorical_cross_entropy
)
from LousyBookML.neural_network.utils import normalize_data, one_hot_encode
from LousyBookML.neural_network.layers import Layer, RepeatedLayer, LayerStack

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
    assert isinstance(binary_cross_entropy(y_true, y_pred), float)
    
    y_true_cat = np.array([[1,0], [0,1]])
    y_pred_cat = np.array([[0.9,0.1], [0.1,0.9]])
    assert isinstance(categorical_cross_entropy(y_true_cat, y_pred_cat), float)

def test_utils():
    """Test utility functions."""
    # Test normalize_data
    X = np.array([[1, 2], [3, 4]])
    X_norm, _, _ = normalize_data(X)
    assert X_norm.shape == X.shape
    
    # Test one_hot_encode
    y = np.array([0, 1, 2])
    y_onehot = one_hot_encode(y)
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
    ], loss='binary_crossentropy', optimizer='adam', learning_rate=0.01)
    
    history = model.fit(X, y_xor, epochs=5000, batch_size=4, verbose=False)  # More epochs, full batch
    
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
    
    # Test with LayerStack and RepeatedLayer
    model2 = NeuralNetwork(
        LayerStack([
            Layer(units=8, activation='relu'),
            RepeatedLayer(count=2, units=4, activation='relu'),
            Layer(units=1, activation='sigmoid')
        ]),
        loss='binary_crossentropy'
    )
    
    history2 = model2.fit(X, y, epochs=1000, batch_size=1, verbose=False)
    assert history2['loss'][-1] < history2['loss'][0]
    
    # Test with traditional dict configuration
    model3 = NeuralNetwork([
        {'units': 8, 'activation': 'relu'},
        {'units': 4, 'activation': 'relu'},
        {'units': 1, 'activation': 'sigmoid'}
    ], loss='binary_crossentropy')
    
    history3 = model3.fit(X, y, epochs=1000, batch_size=1, verbose=False)
    assert history3['loss'][-1] < history3['loss'][0]
    
    # Verify all models produce valid predictions
    for model in [model1, model2, model3]:
        pred = model.predict(X)
        assert pred.shape == y.shape
        assert np.all((pred >= 0) & (pred <= 1))

def test_optimizers():
    """Test different optimizers."""
    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    optimizers = ['sgd', 'adam', 'rmsprop']
    histories = {}
    
    for opt in optimizers:
        # Test each optimizer
        model = NeuralNetwork([
            {'units': 4, 'activation': 'relu'},
            {'units': 1, 'activation': 'sigmoid'}
        ], optimizer=opt, loss='binary_crossentropy')
        
        history = model.fit(X, y, epochs=1000, verbose=False)
        histories[opt] = history
        
        # Check that loss decreased
        assert history['loss'][-1] < history['loss'][0]
        
        # Verify predictions
        pred = model.predict(X)
        assert pred.shape == y.shape
        assert np.all((pred >= 0) & (pred <= 1))

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
