import numpy as np
import pytest
from lousybook01.LousyBookML import LinearRegression

def test_simple_fit():
    """Test fitting on simple linear data."""
    # y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Check coefficients
    assert np.abs(model.coef_[0] - 2.0) < 1e-10
    assert np.abs(model.intercept_ - 1.0) < 1e-10
    assert model.r_squared_ > 0.99

def test_multiple_features():
    """Test with multiple input features."""
    # y = 2x1 + 3x2 + 1
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([6, 11, 16])
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions should be accurate
    predictions = model.predict(np.array([[4, 4]]))
    expected = 2 * 4 + 3 * 4 + 1
    assert np.abs(predictions[0] - expected) < 1e-10

def test_regularization():
    """Test L2 regularization effect."""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    # Without regularization
    model1 = LinearRegression(alpha=0.0)
    model1.fit(X, y)
    
    # With regularization
    model2 = LinearRegression(alpha=1.0)
    model2.fit(X, y)
    
    # Regularized coefficients should be smaller
    assert np.abs(model2.coef_[0]) < np.abs(model1.coef_[0])

def test_input_validation():
    """Test input validation."""
    model = LinearRegression()
    
    # Test invalid shapes
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2]), np.array([[1], [2]]))
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        model.fit(np.array([[1], [2]]), np.array([1, 2, 3]))

def test_prediction_before_fit():
    """Test error when predicting before fitting."""
    model = LinearRegression()
    
    with pytest.raises(RuntimeError):
        model.predict(np.array([1, 2, 3]))

def test_r_squared():
    """Test R-squared calculation."""
    # Perfect fit
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    model = LinearRegression()
    model.fit(X, y)
    
    assert np.abs(model.r_squared_ - 1.0) < 1e-10
    
    # Test score method
    score = model.score(X, y)
    assert np.abs(score - 1.0) < 1e-10

def test_noisy_data():
    """Test with noisy data."""
    np.random.seed(42)
    X = np.array([[x] for x in range(100)])
    y = 2 * X.reshape(-1) + 1 + np.random.normal(0, 0.1, 100)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Check if coefficients are close to true values
    assert np.abs(model.coef_[0] - 2.0) < 0.1
    assert np.abs(model.intercept_ - 1.0) < 0.1
    assert model.r_squared_ > 0.95

def test_get_params():
    """Test parameter getter."""
    model = LinearRegression(alpha=0.1, fit_intercept=True)
    model.fit(np.array([[1], [2]]), np.array([2, 4]))
    
    params = model.get_params()
    assert params['alpha'] == 0.1
    assert params['fit_intercept'] == True
    assert 'coef_' in params
    assert 'intercept_' in params
    assert 'r_squared_' in params
    assert 'mse_' in params
