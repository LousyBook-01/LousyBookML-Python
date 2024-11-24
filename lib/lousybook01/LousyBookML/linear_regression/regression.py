"""
LousyBookML - Linear Regression implementation by LousyBook01
www.youtube.com/@LousyBook01

This module implements linear regression with L2 regularization (Ridge regression).
The implementation supports both normal equation and gradient descent solvers,
making it suitable for both educational purposes and practical use.

Features:
- L2 regularization (Ridge regression) with configurable alpha parameter
- Multiple solvers: normal equation and gradient descent
- Learning rate scheduling for gradient descent
- Built-in R² score calculation for model evaluation
- Automatic bias term handling

Example:
    >>> from LousyBookML import LinearRegression
    >>> model = LinearRegression(alpha=0.1)  # Create model with L2 regularization
    >>> model.fit(X_train, y_train)  # Train the model
    >>> predictions = model.predict(X_test)  # Make predictions
    >>> score = model.score(X_test, y_test)  # Evaluate performance

Made with ❤️ by LousyBook01
"""

import numpy as np
from typing import Optional, Callable, Dict, Any

class LinearRegression:
    """Linear regression with optional L2 regularization (Ridge regression).
    
    This implementation supports both normal equation and gradient descent methods:
    - Normal equation: w = (X^T X + αI)^(-1) X^T y
    - Gradient descent: w = w - lr * ∇J(w)
    where α is the regularization parameter and lr is the learning rate.
    
    Attributes:
        alpha (float): L2 regularization strength (0.0 means no regularization)
        fit_intercept (bool): Whether to fit the intercept (bias term)
        solver (str): Algorithm to use ('normal' or 'gradient_descent')
        learning_rate (float): Initial learning rate for gradient descent
        max_iter (int): Maximum number of iterations for gradient descent
        tol (float): Tolerance for convergence in gradient descent
        learning_rate_schedule (callable): Learning rate schedule function
        weights (np.ndarray): Learned feature weights
        bias (float): Learned bias term
        coef_ (np.ndarray): Learned feature weights (scikit-learn compatibility)
        intercept_ (float): Learned bias term (scikit-learn compatibility)
        r_squared_ (float): R² score of the model
        mse_ (float): Mean squared error of the model
    """
    
    def __init__(self, alpha: float = 0.0, fit_intercept: bool = True,
                 solver: str = 'normal', learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-4,
                 learning_rate_schedule: Optional[Callable[[int], float]] = None):
        """Initialize linear regression model.
        
        Args:
            alpha: L2 regularization coefficient (default: 0.0)
            fit_intercept: Whether to fit the intercept (bias term)
            solver: Algorithm to use ('normal' or 'gradient_descent')
            learning_rate: Initial learning rate for gradient descent
            max_iter: Maximum number of iterations for gradient descent
            tol: Tolerance for convergence in gradient descent
            learning_rate_schedule: Function that takes epoch number and returns learning rate
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate_schedule = learning_rate_schedule or (lambda _: learning_rate)
        
        # Check for very small or large learning rate
        if solver == 'gradient_descent':
            if learning_rate < 1e-5:
                import warnings
                warnings.warn("Learning rate is very small, convergence may be slow")
            elif learning_rate > 1.0:
                import warnings
                warnings.warn("Learning rate is very large, convergence may be unstable")
        
        self.weights = None
        self.bias = None
        self.coef_ = None
        self.intercept_ = None
        self.r_squared_ = None
        self.mse_ = None
        self.n_iter_ = None
        self.loss_curve_ = []
        
    def _check_multicollinearity(self, X: np.ndarray) -> None:
        """Check for multicollinearity in features."""
        if X.shape[1] < 2:
            return
            
        correlations = np.corrcoef(X.T)
        mask = ~np.eye(correlations.shape[0], dtype=bool)
        if np.any(np.abs(correlations[mask]) > 0.9):
            import warnings
            warnings.warn("High multicollinearity detected between features")
            
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0
        prev_loss = float('inf')
        self.n_iter_ = self.max_iter
        self.loss_curve_ = []
        
        # Scale features for better convergence
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X - X_mean) / X_std
        
        # Scale target
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_scaled = (y - y_mean) / y_std
        
        # Initialize weights in scaled space
        weights_scaled = np.zeros(n_features)
        bias_scaled = 0.0 if self.fit_intercept else 0.0
        
        # Normalize learning rate by feature dimension and data size
        base_lr = self.learning_rate * np.sqrt(n_samples) / np.sqrt(n_features)
        
        # Initialize momentum
        momentum = 0.9
        velocity_w = np.zeros_like(weights_scaled)
        velocity_b = 0.0
        
        for epoch in range(self.max_iter):
            # Get current learning rate
            lr = self.learning_rate_schedule(epoch) * base_lr
            
            # Compute predictions and error
            if self.fit_intercept:
                y_pred = X_scaled @ weights_scaled + bias_scaled
            else:
                y_pred = X_scaled @ weights_scaled
                
            error = y_pred - y_scaled
            
            # Compute gradients with L2 regularization
            gradients = (X_scaled.T @ error) / n_samples + self.alpha * weights_scaled
            
            # Update with momentum
            velocity_w = momentum * velocity_w - lr * gradients
            weights_scaled += velocity_w
            
            if self.fit_intercept:
                bias_grad = np.mean(error)
                velocity_b = momentum * velocity_b - lr * bias_grad
                bias_scaled += velocity_b
            
            # Compute loss
            current_loss = np.mean(error ** 2) + 0.5 * self.alpha * np.sum(weights_scaled ** 2)
            current_loss = float(current_loss)
            self.loss_curve_.append(current_loss)
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tol:
                self.n_iter_ = epoch + 1
                break
            prev_loss = current_loss
            
        # Unscale weights and compute intercept
        self.weights = weights_scaled * y_std / X_std
        if self.fit_intercept:
            self.bias = y_mean - np.sum(X_mean * self.weights)
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the linear regression model.
        
        Args:
            X: Input features matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Raises:
            ValueError: If solver is not recognized
            Warning: If constant features are detected
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D array")
            
        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError("y must be a 1-D array or a 2-D array with shape (n_samples, 1)")
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError("y must be a 1-D array")
            
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        n_samples, n_features = X.shape
        
        # Input validation
        if n_samples < n_features:
            import warnings
            warnings.warn("Number of samples is less than number of features")
            
        # Check for constant features and multicollinearity
        constant_mask = np.all(X == X[0, :], axis=0)
        if np.any(constant_mask):
            import warnings
            warnings.warn("Constant features detected in X")
            
        self._check_multicollinearity(X)
            
        if self.solver == 'normal':
            # Scale features for numerical stability
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1
            X_scaled = (X - X_mean) / X_std
            
            # Scale target
            y_mean = np.mean(y)
            y_std = np.std(y)
            y_scaled = (y - y_mean) / y_std
            
            try:
                # Add bias term if needed
                if self.fit_intercept:
                    X_bias = np.c_[np.ones(n_samples), X_scaled]
                else:
                    X_bias = X_scaled
                    
                # Compute weights using normal equation with regularization
                n_total = X_bias.shape[1]
                reg_matrix = self.alpha * np.eye(n_total)
                if self.fit_intercept:
                    reg_matrix[0, 0] = 0  # Don't regularize bias term
                    
                # Add small constant to diagonal for numerical stability
                reg_matrix += 1e-10 * np.eye(n_total)
                
                # Solve normal equation
                weights = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y_scaled)
                
                # Unscale weights and compute intercept
                if self.fit_intercept:
                    self.weights = weights[1:] * y_std / X_std
                    self.bias = y_mean - np.sum(X_mean * self.weights)
                else:
                    self.weights = weights * y_std / X_std
                    self.bias = 0.0
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("Singular matrix: try increasing alpha for regularization")
        elif self.solver == 'gradient_descent':
            # Check learning rate
            if self.learning_rate < 1e-5:
                import warnings
                warnings.warn("Learning rate is very small, convergence may be slow")
            elif self.learning_rate > 1.0:
                import warnings
                warnings.warn("Learning rate is very large, convergence may be unstable")
                
            self._gradient_descent(X, y)
        else:
            raise ValueError(f"Solver '{self.solver}' not recognized. Use 'normal' or 'gradient_descent'")
            
        # Set scikit-learn compatible attributes
        self.coef_ = self.weights
        self.intercept_ = self.bias
        
        # Compute R² score and MSE
        y_pred = self.predict(X)
        residuals = y - y_pred
        self.mse_ = np.mean(residuals ** 2)
        total_ss = np.sum((y - np.mean(y)) ** 2)
        if total_ss == 0:
            self.r_squared_ = 1.0  # Perfect fit for constant target
        else:
            self.r_squared_ = 1 - np.sum(residuals ** 2) / total_ss
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input features.
        
        Args:
            X: Input features matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
            
        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before making predictions.")
            
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return X @ self.weights + self.bias
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score (coefficient of determination).
        
        Args:
            X: Input features matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            R² score (1.0 is perfect prediction)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
            
        y_pred = self.predict(X)
        residuals = y - y_pred
        total_ss = np.sum((y - np.mean(y)) ** 2)
        if total_ss == 0:
            return 1.0  # Perfect fit for constant target
        else:
            return 1 - np.sum(residuals ** 2) / total_ss
        
    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Returns:
            Parameter names mapped to their values.
        """
        params = {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'solver': self.solver,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol
        }
        
        # Add fitted parameters if available
        if hasattr(self, 'coef_'):
            params['coef_'] = self.coef_
        if hasattr(self, 'intercept_'):
            params['intercept_'] = self.intercept_
        if hasattr(self, 'r_squared_'):
            params['r_squared_'] = self.r_squared_
        if hasattr(self, 'mse_'):
            params['mse_'] = self.mse_
            
        return params
