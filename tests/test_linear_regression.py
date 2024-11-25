"""Test cases for the Linear Regression module."""

import numpy as np
import pytest
from LousyBookML import LinearRegression

# ================ Test Losses ================

def test_mean_squared_error():
    """Test MSE calculation."""
    y_true = np.array([[1], [2], [3], [4], [5]])
    y_pred = np.array([[1.1], [2.1], [3.1], [4.1], [5.1]])
    
    mse = LinearRegression.mean_squared_error(y_true, y_pred)
    assert isinstance(mse, float)
    assert mse == pytest.approx(0.01, rel=1e-6)
    
    # Test with perfect predictions
    assert LinearRegression.mean_squared_error(y_true, y_true) == 0.0

def test_mean_absolute_error():
    """Test MAE calculation."""
    y_true = np.array([[1], [2], [3], [4], [5]])
    y_pred = np.array([[1.1], [2.1], [3.1], [4.1], [5.1]])
    
    mae = LinearRegression.mean_absolute_error(y_true, y_pred)
    assert isinstance(mae, float)
    assert mae == pytest.approx(0.1, rel=1e-6)
    
    # Test with perfect predictions
    assert LinearRegression.mean_absolute_error(y_true, y_true) == 0.0

def test_r2_score():
    """Test R² score calculation."""
    y_true = np.array([[1], [2], [3], [4], [5]])
    y_pred = np.array([[1.1], [2.1], [3.1], [4.1], [5.1]])
    
    r2 = LinearRegression.r2_score(y_true, y_pred)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1
    
    # Test with perfect predictions
    assert LinearRegression.r2_score(y_true, y_true) == 1.0
    
    # Test with mean predictions (should give R² = 0)
    y_mean = np.full_like(y_true, y_true.mean())
    assert LinearRegression.r2_score(y_true, y_mean) == pytest.approx(0.0, abs=1e-6)

# ================ Test Utils ================

def test_standardize_data():
    """Test data standardization."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_std, mean, std = LinearRegression.standardize_data(X)
    
    assert X_std.shape == X.shape
    assert np.allclose(np.mean(X_std, axis=0), [0, 0])
    assert np.allclose(np.std(X_std, axis=0), [1, 1])
    
    # Test with constant feature (std = 0)
    X_const = np.array([[1, 2], [1, 4], [1, 6]])
    X_const_std, _, _ = LinearRegression.standardize_data(X_const)
    assert not np.any(np.isnan(X_const_std))
    assert np.allclose(X_const_std[:, 0], 0)

def test_add_polynomial_features():
    """Test polynomial feature generation."""
    X = np.array([[1], [2], [3]])
    
    # Test degree 2
    X_poly = LinearRegression.add_polynomial_features(X, degree=2)
    assert X_poly.shape == (3, 2)
    assert np.array_equal(X_poly, np.array([[1, 1], [2, 4], [3, 9]]))
    
    # Test degree 3
    X_poly = LinearRegression.add_polynomial_features(X, degree=3)
    assert X_poly.shape == (3, 3)
    assert np.array_equal(X_poly, np.array([[1, 1, 1], [2, 4, 8], [3, 9, 27]]))
    
    # Test with 2D input
    X = np.array([[1, 2], [3, 4]])
    X_poly = LinearRegression.add_polynomial_features(X, degree=2)
    assert X_poly.shape == (2, 4)  # Original features + squared terms

# ================ Test Model ================

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 1)  # Single feature for simplicity
    y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1))  # y = 2x + 1 + noise
    return X, y

def test_model_initialization():
    """Test model initialization."""
    # Test default initialization
    model = LinearRegression()
    assert not model.fitted
    assert model.method == 'normal_equation'
    assert model.learning_rate == 0.01
    assert model.n_iterations == 1000
    assert model.standardize == True
    
    # Test invalid method
    with pytest.raises(ValueError, match="Method must be 'normal_equation' or 'gradient_descent'"):
        LinearRegression(method='invalid_method')
    
    # Test invalid learning rate
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        LinearRegression(learning_rate=-1)

def test_model_fit(sample_data):
    """Test model fitting."""
    X, y = sample_data
    
    # Test normal equation method
    model = LinearRegression(method='normal_equation')
    history = model.fit(X, y)
    assert model.fitted
    assert isinstance(history, dict)
    assert 'loss' in history
    
    # Check if coefficients are close to true values (2x + 1)
    assert abs(model.weights[0] - 2.0) < 0.1
    assert abs(model.bias - 1.0) < 0.1
    
    # Test predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert LinearRegression.r2_score(y, y_pred) > 0.95

def test_gradient_descent(sample_data):
    """Test gradient descent optimization."""
    X, y = sample_data
    
    model = LinearRegression(
        method='gradient_descent',
        learning_rate=0.01,
        n_iterations=1000
    )
    history = model.fit(X, y)
    assert model.fitted
    assert isinstance(history, dict)
    assert 'loss' in history
    assert len(history['loss']) == model.n_iterations
    assert history['loss'][0] > history['loss'][-1]  # Loss should decrease
    
    # Check if coefficients are close to true values (2x + 1)
    assert abs(model.weights[0] - 2.0) < 0.2  # Slightly larger tolerance for GD
    assert abs(model.bias - 1.0) < 0.2
    
    # Test predictions
    y_pred = model.predict(X)
    assert LinearRegression.r2_score(y, y_pred) > 0.95

def test_input_validation():
    """Test input validation."""
    model = LinearRegression()
    
    # Test invalid shapes
    with pytest.raises(ValueError, match="Found input samples: 1, but got target samples: 2"):
        model.fit(np.array([[1]]), np.array([[1], [2]]))
    
    # Test prediction without fitting
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        model.predict(np.array([[1]]))
    
    # Test prediction with wrong feature count
    model.fit(np.array([[1], [2]]), np.array([[1], [2]]))
    with pytest.raises(ValueError, match="Expected 1 features but got 2"):
        model.predict(np.array([[1, 2]]))
