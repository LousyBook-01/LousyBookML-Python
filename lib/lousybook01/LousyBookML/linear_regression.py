"""
Linear Regression Implementation from scratch by LousyBook01.

A robust implementation of linear regression that supports multiple features,
regularization, and provides various statistical metrics for model evaluation.

Features:
    - Multi-feature support with automatic input validation
    - L2 (Ridge) regularization for preventing overfitting
    - Statistical metrics (R-squared, MSE) for model evaluation
    - Robust numerical computations using numpy
    - Input validation and automatic reshaping
    - Flexible model configuration with intercept fitting
    - Efficient matrix operations for large datasets
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable

class LinearRegression:
    """Linear Regression with optional regularization and statistical metrics.
    
    This implementation uses the normal equation method to find the optimal
    parameters for linear regression. It supports L2 regularization (Ridge regression)
    and provides various metrics to evaluate model performance.
    
    The model fits a line to the data by minimizing the sum of squared residuals:
        min_w ||y - Xw||^2 + alpha * ||w||^2
    where w are the model parameters, X is the feature matrix, y are the targets,
    and alpha is the regularization strength.
    
    Args:
        alpha (float, optional): Regularization strength (L2 regularization).
            Higher values mean stronger regularization. Defaults to 0.0.
        fit_intercept (bool, optional): Whether to calculate and include the
            intercept term. Defaults to True.
        solver (str, optional): Algorithm to use for optimization.
            Options: 'normal' (normal equation) or 'gradient_descent'.
            Defaults to 'normal'.
        learning_rate (float, optional): Initial learning rate for gradient descent.
            Only used when solver='gradient_descent'. Defaults to 0.01.
        max_iter (int, optional): Maximum number of iterations for gradient descent.
            Only used when solver='gradient_descent'. Defaults to 1000.
        tol (float, optional): Tolerance for convergence.
            Only used when solver='gradient_descent'. Defaults to 1e-4.
        learning_rate_schedule (callable, optional): Function that takes the epoch
            number and returns the learning rate. If None, uses constant learning rate.
    
    Attributes:
        coef_ (numpy.ndarray): Estimated coefficients for the features.
            Shape is (n_features,).
        intercept_ (float): Independent term in the linear model (y-intercept).
            Only present if fit_intercept=True.
        r_squared_ (float): R-squared score of the model, representing the
            proportion of variance in the target that's predictable from the features.
        mse_ (float): Mean squared error of the model on the training data.
        n_iter_ (int): Number of iterations performed by the solver.
        loss_curve_ (list): List of convergence values at each iteration.
    
    Example:
        >>> model = LinearRegression(alpha=0.1)
        >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        >>> y = np.dot(X, np.array([1, 2])) + 3
        >>> model.fit(X, y)
        >>> print(f"Coefficients: {model.coef_}")
        >>> print(f"Intercept: {model.intercept_}")
        >>> print(f"R-squared: {model.r_squared_}")
    """
    
    def __init__(self, alpha: float = 0.0, fit_intercept: bool = True,
                 solver: str = 'normal', learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-4,
                 learning_rate_schedule: Optional[Callable[[int], float]] = None):
        """Initialize LinearRegression.
        
        Args:
            alpha (float, optional): Regularization strength (L2 regularization).
                Higher values mean stronger regularization. Defaults to 0.0.
            fit_intercept (bool, optional): Whether to calculate and include the
                intercept term. Defaults to True.
            solver (str, optional): Algorithm to use for optimization.
                Options: 'normal' (normal equation) or 'gradient_descent'.
                Defaults to 'normal'.
            learning_rate (float, optional): Initial learning rate for gradient descent.
                Only used when solver='gradient_descent'. Defaults to 0.01.
            max_iter (int, optional): Maximum number of iterations for gradient descent.
                Only used when solver='gradient_descent'. Defaults to 1000.
            tol (float, optional): Tolerance for convergence.
                Only used when solver='gradient_descent'. Defaults to 1e-4.
            learning_rate_schedule (callable, optional): Function that takes the epoch
                number and returns the learning rate. If None, uses constant learning rate.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver.lower()
        if self.solver not in ['normal', 'gradient_descent']:
            raise ValueError("solver must be 'normal' or 'gradient_descent'")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate_schedule = learning_rate_schedule
        
        self.coef_ = None
        self.intercept_ = None
        self.r_squared_ = None
        self.mse_ = None
        self.n_iter_ = None
        self.loss_curve_ = []
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and format input data.
        
        Performs several checks and transformations on the input data:
        1. Converts inputs to numpy arrays if needed
        2. Ensures correct dimensionality
        3. Checks for consistent sample sizes
        4. Handles 1D input arrays
        
        Args:
            X (numpy.ndarray): Input features. Can be shape (n_samples, n_features)
                or (n_samples,) for single-feature data.
            y (numpy.ndarray): Target values of shape (n_samples,)
        
        Returns:
            tuple: Tuple containing:
                - X (numpy.ndarray): Validated and reshaped feature matrix
                - y (numpy.ndarray): Validated and reshaped target vector
        
        Raises:
            ValueError: If input dimensions are inconsistent or invalid
        """
        # Convert to numpy arrays if not already
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Reshape X if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check dimensions
        if X.ndim != 2:
            raise ValueError("X must be 2D array or 1D array, got shape {0}".format(X.shape))
        if y.ndim != 1:
            raise ValueError("y must be 1D array, got shape {0}".format(y.shape))
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
            
        return X, y
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve linear regression using gradient descent.
        
        Args:
            X (numpy.ndarray): Feature matrix with shape (n_samples, n_features)
            y (numpy.ndarray): Target values with shape (n_samples,)
            
        Returns:
            numpy.ndarray: Optimal parameters
        """
        n_samples = X.shape[0]
        
        # Initialize parameters with small random values for better numerical stability
        np.random.seed(42)  # For reproducibility
        beta = np.random.randn(X.shape[1]) * 0.01
        
        # Scale features for better numerical stability
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_scaled = (X - X_mean) / X_std
        
        # Scale target for better numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_scaled = (y - y_mean) / y_std
        
        prev_loss = float('inf')
        self.loss_curve_ = []
        
        # Increase max_iter for better convergence with learning rate scheduling
        actual_max_iter = self.max_iter * 5 if self.learning_rate_schedule else self.max_iter
        
        # Initial learning rate boost for faster convergence
        lr_boost = 5.0 if self.learning_rate_schedule else 1.0
        
        for epoch in range(actual_max_iter):
            # Get current learning rate with initial boost
            current_lr = (self.learning_rate_schedule(epoch // 5) * lr_boost
                        if self.learning_rate_schedule 
                        else self.learning_rate)
            
            # Gradually reduce learning rate boost
            if epoch > actual_max_iter // 2:
                lr_boost = max(1.0, lr_boost * 0.99)
            
            # Compute predictions and error
            y_pred = X_scaled @ beta
            error = y_pred - y_scaled
            
            # Compute loss
            current_loss = np.mean(error**2)
            if self.alpha > 0:
                current_loss += self.alpha * np.sum(beta[1:]**2 if self.fit_intercept else beta**2)
            
            self.loss_curve_.append(current_loss)
            
            # Compute gradients (with regularization)
            gradients = 2 * X_scaled.T @ error / n_samples
            if self.alpha > 0:
                reg_gradients = 2 * self.alpha * beta
                if self.fit_intercept:
                    reg_gradients[0] = 0  # Don't regularize intercept
                gradients += reg_gradients
            
            # Update parameters with gradient clipping for stability
            gradients = np.clip(gradients, -1e10, 1e10)  # Prevent exploding gradients
            beta -= current_lr * gradients
            
            # Check for convergence
            if abs(prev_loss - current_loss) < self.tol:
                self.n_iter_ = epoch + 1
                break
                
            prev_loss = current_loss
        else:
            self.n_iter_ = actual_max_iter
        
        # Unscale parameters to get original scale coefficients
        beta = beta * y_std / X_std
        if self.fit_intercept:
            beta[0] = y_mean - np.sum(beta[1:] * X_mean[1:])
            
        return beta
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """Fit linear regression model.
        
        Args:
            X (numpy.ndarray): Training data of shape (n_samples, n_features)
                or (n_samples,) for single-feature data.
            y (numpy.ndarray): Target values of shape (n_samples,)
        
        Returns:
            self: Returns an instance of self.
        """
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape
        
        # Check for constant features
        constant_mask = np.all(X == X[0, :], axis=0)
        if np.any(constant_mask):
            import warnings
            warnings.warn("X contains constant features. This may cause numerical instability.", UserWarning)
        
        # Check for multicollinearity
        if n_features > 1:
            corr_matrix = np.corrcoef(X.T)
            if np.any(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]) > 0.9999):
                import warnings
                warnings.warn("Perfect multicollinearity detected in features.", UserWarning)
        
        # Check for very small or large learning rate in gradient descent
        if self.solver == 'gradient_descent':
            import warnings
            if self.learning_rate < 1e-6:
                warnings.warn("Very small learning rate may cause slow convergence.", UserWarning)
            elif self.learning_rate > 1.0:
                warnings.warn("Very large learning rate may cause divergence.", UserWarning)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])
            n_features += 1
        
        # Solve using specified method
        if self.solver == 'normal':
            # Solve normal equations with regularization
            # (X^T X + αI)β = X^T y
            if self.alpha > 0:
                reg_matrix = self.alpha * np.eye(n_features)
                if self.fit_intercept:
                    reg_matrix[0, 0] = 0  # Don't regularize intercept
            else:
                reg_matrix = 0
                
            try:
                beta = np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)
            except np.linalg.LinAlgError:
                import warnings
                warnings.warn("Matrix is ill-conditioned. Using pseudo-inverse for solution.", UserWarning)
                beta = np.linalg.pinv(X.T @ X + reg_matrix) @ X.T @ y
            
            self.n_iter_ = 1
            self.loss_curve_ = [np.inf]
            
        else:  # gradient_descent
            beta = self._gradient_descent(X, y)
        
        # Extract intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = beta
        
        # Calculate fit metrics
        y_pred = self.predict(X[:, 1:] if self.fit_intercept else X)
        self.mse_ = np.mean((y - y_pred) ** 2)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        self.r_squared_ = 1 - (ss_res / ss_tot)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear regression model.
        
        Makes predictions on new data using the fitted model.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
                or (n_samples,) for single-feature data.
        
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"X has {X.shape[1]} features, but model was trained with {len(self.coef_)} features")
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R-squared score of the model.
        
        Evaluates the model on the given data and returns the R-squared score.
        
        Args:
            X (numpy.ndarray): Test samples of shape (n_samples, n_features)
                or (n_samples,) for single-feature data.
            y (numpy.ndarray): True values for X of shape (n_samples,)
        
        Returns:
            float: R-squared score
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters.
        
        Returns the model's parameters, including coefficients, intercept, and
        regularization strength.
        
        Returns:
            dict: Model parameters
        """
        return {
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'r_squared_': self.r_squared_,
            'mse_': self.mse_,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'solver': self.solver,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'n_iter_': self.n_iter_,
            'loss_curve_': self.loss_curve_
        }
