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

def test_cross_validation():
    """Test k-fold cross-validation."""
    np.random.seed(42)
    X = np.array([[x] for x in range(100)])
    y = 2 * X.reshape(-1) + 1 + np.random.normal(0, 0.1, 100)
    
    model = LinearRegression()
    
    # Implement k-fold cross-validation
    k = 5
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        # Create train-test split
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        X_train = np.concatenate([X[:test_start], X[test_end:]])
        y_train = np.concatenate([y[:test_start], y[test_end:]])
        
        # Fit and evaluate
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    
    # Check if all folds perform well
    assert np.mean(scores) > 0.95
    assert np.std(scores) < 0.1

def test_feature_scaling():
    """Test with and without feature scaling."""
    # Generate data with different scales
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X[:, 1] = X[:, 1] * 1000  # Make second feature much larger
    y = X[:, 0] + 0.001 * X[:, 1] + np.random.normal(0, 0.1, 100)
    
    # Without scaling
    model1 = LinearRegression()
    model1.fit(X, y)
    score1 = model1.score(X, y)
    
    # With scaling
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    model2 = LinearRegression()
    model2.fit(X_scaled, y)
    score2 = model2.score(X_scaled, y)
    
    # Scaled version should perform better or similarly
    assert score2 >= score1 * 0.95

def test_polynomial_features():
    """Test with polynomial features."""
    # Generate non-linear data
    X = np.array([[x] for x in np.linspace(-5, 5, 100)])
    y = 1 + 2*X.reshape(-1) + 3*X.reshape(-1)**2 + np.random.normal(0, 0.1, 100)
    
    # Create polynomial features
    X_poly = np.column_stack([X, X**2])
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Model should capture the quadratic relationship
    assert model.r_squared_ > 0.95
    assert np.abs(model.coef_[1] - 3.0) < 0.5  # Coefficient for x^2
    assert np.abs(model.coef_[0] - 2.0) < 0.5  # Coefficient for x

def test_gradient_descent_vs_normal():
    """Compare gradient descent with normal equation."""
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    true_coeffs = np.array([1, -2, 3, -4, 5])
    y = np.dot(X, true_coeffs) + np.random.normal(0, 0.1, 1000)
    
    # Fit with normal equation
    model_normal = LinearRegression(solver='normal')
    model_normal.fit(X, y)
    
    # Fit with gradient descent
    model_gd = LinearRegression(solver='gradient_descent', learning_rate=0.01, max_iter=1000)
    model_gd.fit(X, y)
    
    # Both methods should achieve similar R-squared
    assert np.abs(model_normal.r_squared_ - model_gd.r_squared_) < 0.01
    
    # Coefficients should be similar
    assert np.allclose(model_normal.coef_, model_gd.coef_, rtol=0.1)

def test_learning_rate_scheduling():
    """Test learning rate scheduling."""
    X = np.array([[x] for x in range(100)])
    y = 2 * X.reshape(-1) + 1
    
    # Test different learning rate schedules
    schedules = {
        'constant': lambda epoch: 0.01,
        'time_decay': lambda epoch: 0.01 / (1 + 0.1 * epoch),
        'exponential': lambda epoch: 0.01 * (0.95 ** epoch)
    }
    
    for schedule_name, schedule_fn in schedules.items():
        model = LinearRegression(
            solver='gradient_descent',
            learning_rate_schedule=schedule_fn,
            max_iter=100
        )
        model.fit(X, y)
        
        # All schedules should converge
        assert model.r_squared_ > 0.99

def test_convergence_monitoring():
    """Test convergence monitoring."""
    X = np.array([[x] for x in range(100)])
    y = 2 * X.reshape(-1) + 1
    
    model = LinearRegression(
        solver='gradient_descent',
        tol=1e-6,
        max_iter=1000
    )
    model.fit(X, y)
    
    # Check if convergence is monitored
    assert hasattr(model, 'n_iter_')  # Number of iterations
    assert hasattr(model, 'loss_curve_')  # Loss history
    
    # Loss should decrease
    assert model.loss_curve_[-1] < model.loss_curve_[0]
    
    # Should converge before max_iter
    assert model.n_iter_ < 1000

def test_edge_cases():
    """Test edge cases and error handling."""
    model = LinearRegression()
    
    # Test with constant feature
    X = np.ones((10, 1))
    y = np.random.randn(10)
    with pytest.warns(UserWarning):
        model.fit(X, y)
    
    # Test with perfect multicollinearity
    X = np.array([[1, 2], [2, 4], [3, 6]])  # Second column is 2 * first column
    y = np.array([1, 2, 3])
    with pytest.warns(UserWarning):
        model.fit(X, y)
    
    # Test with very small learning rate
    model = LinearRegression(solver='gradient_descent', learning_rate=1e-10)
    with pytest.warns(UserWarning):
        model.fit(np.array([[1], [2]]), np.array([1, 2]))
    
    # Test with very large learning rate
    model = LinearRegression(solver='gradient_descent', learning_rate=1e10)
    with pytest.warns(UserWarning):
        model.fit(np.array([[1], [2]]), np.array([1, 2]))
