import numpy as np
import matplotlib.pyplot as plt
from lousybook01.LousyBookML import LinearRegression

def simple_regression_example():
    """Demonstrate simple linear regression with visualization."""
    # Generate sample data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.reshape(-1) + 1 + np.random.normal(0, 0.5, 100)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Print results
    print("\nSimple Linear Regression Results:")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R-squared: {model.r_squared_:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.title('Simple Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def multiple_regression_example():
    """Demonstrate multiple linear regression."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)  # 3 features
    true_coef = [2, -1, 0.5]
    y = np.dot(X, true_coef) + np.random.normal(0, 0.1, n_samples)
    
    # Create and train model with regularization
    model = LinearRegression(alpha=0.1)  # With L2 regularization
    model.fit(X, y)
    
    # Print results
    print("\nMultiple Linear Regression Results:")
    print("True coefficients:", true_coef)
    print("Estimated coefficients:", model.coef_)
    print(f"R-squared: {model.r_squared_:.4f}")
    print(f"Mean Squared Error: {model.mse_:.4f}")

def polynomial_features_example():
    """Demonstrate polynomial regression using linear regression."""
    # Generate sample data
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = 0.5 * X.reshape(-1)**2 + X.reshape(-1) + 2 + np.random.normal(0, 1, 100)
    
    # Create polynomial features
    X_poly = np.column_stack([X, X**2])
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Make predictions
    y_pred = model.predict(X_poly)
    
    # Print results
    print("\nPolynomial Regression Results:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R-squared: {model.r_squared_:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    plt.plot(X, y_pred, color='red', label='Polynomial fit')
    plt.title('Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def regularization_comparison():
    """Compare models with different regularization strengths."""
    # Generate noisy data
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 10)  # 10 features
    true_coef = np.array([1, 0.5, 0.2, 0.1, 0.05, 0, 0, 0, 0, 0])
    y = np.dot(X, true_coef) + np.random.normal(0, 0.1, n_samples)
    
    # Try different regularization strengths
    alphas = [0, 0.1, 1.0, 10.0]
    
    print("\nRegularization Comparison:")
    for alpha in alphas:
        model = LinearRegression(alpha=alpha)
        model.fit(X, y)
        print(f"\nAlpha = {alpha}")
        print(f"Coefficients: {model.coef_}")
        print(f"R-squared: {model.r_squared_:.4f}")
        print(f"MSE: {model.mse_:.4f}")

if __name__ == "__main__":
    print("Running Linear Regression Examples...")
    
    simple_regression_example()
    multiple_regression_example()
    polynomial_features_example()
    regularization_comparison()
    
    print("\nAll examples completed!")
