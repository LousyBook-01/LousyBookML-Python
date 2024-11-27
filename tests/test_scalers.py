"""Test cases for the Data Scaling module."""

import numpy as np
import pytest
from LousyBookML.scalers import StandardScaler, MinMaxScaler, RobustScaler

def test_standard_scaler():
    """Test StandardScaler with various input types and edge cases."""
    # Test basic scaling
    X = np.array([[1., -1., 2.], 
                  [2., 0., 0.], 
                  [0., 1., -1.]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check mean and unit variance
    assert np.allclose(X_scaled.mean(axis=0), 0.0)
    assert np.allclose(X_scaled.std(axis=0), 1.0)
    
    # Test inverse transform
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inverse)
    
    # Test with constant feature
    X_constant = np.array([[1., 2., 3.],
                          [1., 4., 6.],
                          [1., 6., 9.]])
    scaler.fit(X_constant)
    # First feature should not be scaled (std = 0)
    assert scaler.scale_[0] == 1.0
    
    # Test with single feature
    X_1d = np.array([1., 2., 3.])
    scaler = StandardScaler()
    X_1d_scaled = scaler.fit_transform(X_1d)
    assert X_1d_scaled.shape == X_1d.shape
    
    # Test error cases
    with pytest.raises(ValueError):
        scaler.transform(None)
    with pytest.raises(ValueError):
        scaler = StandardScaler()
        scaler.transform(X)  # Not fitted yet

def test_minmax_scaler():
    """Test MinMaxScaler with various input types and edge cases."""
    # Test basic scaling to [0,1]
    X = np.array([[1., -1., 2.], 
                  [2., 0., 0.], 
                  [0., 1., -1.]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check bounds
    assert np.all(X_scaled >= 0.0)
    assert np.all(X_scaled <= 1.0)
    
    # Test custom feature range (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    assert np.all(X_scaled >= -1.0)
    assert np.all(X_scaled <= 1.0)
    
    # Test inverse transform
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inverse)
    
    # Test with constant feature
    X_constant = np.array([[1., 2.],
                          [1., 4.],
                          [1., 6.]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_constant)
    # First feature should be all zeros (max = min)
    assert np.allclose(X_scaled[:, 0], 0.0)
    
    # Test error cases
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))  # Invalid range
    with pytest.raises(ValueError):
        scaler = MinMaxScaler()
        scaler.transform(X)  # Not fitted yet

def test_robust_scaler():
    """Test RobustScaler with various input types and edge cases."""
    # Generate data with outliers
    X = np.array([[1., -1., 2.], 
                  [2., 0., 0.], 
                  [0., 1., -1.],
                  [100., -100., 200.]])  # Outliers
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check that outliers don't dominate the scaling
    assert np.all(np.abs(X_scaled[:-1]) < np.abs(X_scaled[-1]))
    
    # Test inverse transform
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inverse)
    
    # Test with custom quantile range
    scaler = RobustScaler(quantile_range=(10, 90))
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape
    
    # Test with centering only
    scaler = RobustScaler(with_scaling=False)
    X_scaled = scaler.fit_transform(X)
    # Check that only centering was applied
    assert np.allclose(scaler.scale_, 1.0)
    
    # Test with scaling only
    scaler = RobustScaler(with_centering=False)
    X_scaled = scaler.fit_transform(X)
    # Check that no centering was applied
    assert np.allclose(scaler.center_, 0.0)
    
    # Test error cases
    with pytest.raises(ValueError):
        RobustScaler(quantile_range=(50, 25))  # Invalid range
    with pytest.raises(ValueError):
        scaler = RobustScaler()
        scaler.transform(X)  # Not fitted yet

def test_scaler_edge_cases():
    """Test all scalers with edge cases and special inputs."""
    scalers = [
        StandardScaler(),
        MinMaxScaler(),
        RobustScaler()
    ]
    
    # Test empty arrays
    for scaler in scalers:
        with pytest.raises(ValueError):
            scaler.fit(np.array([]))
            
    # Test single value
    X_single = np.array([[1.0]])
    for scaler in scalers:
        X_scaled = scaler.fit_transform(X_single)
        assert X_scaled.shape == X_single.shape
        
    # Test arrays with infinity and NaN
    X_inf = np.array([[1., np.inf], [3., 4.]])
    X_nan = np.array([[1., np.nan], [3., 4.]])
    for scaler in scalers:
        with pytest.raises(ValueError):
            scaler.fit(X_inf)
        with pytest.raises(ValueError):
            scaler.fit(X_nan)
            
    # Test copy parameter
    X = np.array([[1., 2.], [3., 4.]])
    for scaler in scalers:
        # With copy=True
        X_orig = X.copy()
        scaler = type(scaler)(copy=True)
        X_scaled = scaler.fit_transform(X)
        assert not np.array_equal(X, X_scaled)  # Original should be unchanged
        assert np.array_equal(X, X_orig)
        
        # With copy=False
        scaler = type(scaler)(copy=False)
        X_orig = X.copy()
        X_scaled = scaler.fit_transform(X_orig)
        assert np.array_equal(X_orig, X_scaled)  # Original should be modified
