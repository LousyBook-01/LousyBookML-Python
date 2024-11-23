import numpy as np
import matplotlib.pyplot as plt
from lousybook01.LousyBookML import NeuralNetwork, LayerConfig

def test_xor_generalization():
    """Test XOR with noisy inputs."""
    print("\nTesting XOR Generalization:")
    
    # Training data (clean XOR)
    X_train = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1]])
    y_train = np.array([[0, 1, 1, 0]])
    
    # Create network
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=4, activation='leaky_relu', initialization='he'),
        LayerConfig(size=1, activation='sigmoid', initialization='xavier')
    ]
    
    model = NeuralNetwork(
        layer_configs=layers,
        optimizer='rmsprop',
        learning_rate=0.1,
        momentum_beta=0.9,
        batch_norm=True
    )
    
    # Train model
    print("Training on clean XOR data...")
    model.train(X_train, y_train, epochs=1000, batch_size=4)
    
    # Generate noisy test data
    np.random.seed(42)
    num_test = 100
    noise_level = 0.1
    
    X_test = []
    y_test = []
    
    for _ in range(num_test):
        # Randomly select one of the four XOR input patterns
        idx = np.random.randint(0, 4)
        x = X_train[:, idx:idx+1] + np.random.normal(0, noise_level, (2, 1))
        y = y_train[:, idx:idx+1]
        
        X_test.append(x)
        y_test.append(y)
    
    X_test = np.hstack(X_test)
    y_test = np.hstack(y_test)
    
    # Test predictions
    predictions = model.predict(X_test)
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == y_test)
    
    print(f"\nResults on {num_test} noisy test samples:")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Visualize decision boundary
    plt.figure(figsize=(10, 8))
    
    # Create a grid of points
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.array([X1.ravel(), X2.ravel()])
    
    # Get predictions for grid points
    Z = model.predict(grid_points)
    Z = Z.reshape(X1.shape)
    
    # Plot decision boundary
    plt.contourf(X1, X2, Z, alpha=0.4, levels=np.linspace(0, 1, 21))
    plt.colorbar(label='Prediction')
    
    # Plot training points
    plt.scatter(X_train[0, y_train[0] == 0], X_train[1, y_train[0] == 0], 
               c='red', marker='o', label='Train (0)', s=100)
    plt.scatter(X_train[0, y_train[0] == 1], X_train[1, y_train[0] == 1], 
               c='blue', marker='o', label='Train (1)', s=100)
    
    # Plot test points
    plt.scatter(X_test[0, y_test[0] == 0], X_test[1, y_test[0] == 0], 
               c='red', marker='.', alpha=0.5, label='Test (0)')
    plt.scatter(X_test[0, y_test[0] == 1], X_test[1, y_test[0] == 1], 
               c='blue', marker='.', alpha=0.5, label='Test (1)')
    
    plt.title('XOR Decision Boundary with Noisy Test Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_spiral_classification():
    """Test on spiral classification problem."""
    print("\nTesting Spiral Classification:")
    
    # Generate spiral data
    def generate_spiral_data(points_per_class, classes, noise=0.1):
        X = []
        y = []
        for class_idx in range(classes):
            r = np.linspace(0.0, 1, points_per_class)
            t = np.linspace(class_idx * 4, (class_idx + 1) * 4, points_per_class) + \
                np.random.randn(points_per_class) * noise
            X.append(np.column_stack((r * np.sin(t * 2.5), r * np.cos(t * 2.5))))
            y.append(np.zeros((points_per_class, classes)))
            y[-1][:, class_idx] = 1
        return np.vstack(X).T, np.vstack(y).T
    
    # Generate training and test data
    np.random.seed(42)
    X_train, y_train = generate_spiral_data(200, 2, noise=0.1)  # More training points
    X_test, y_test = generate_spiral_data(50, 2, noise=0.15)
    
    # Create model with deeper architecture
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=32, activation='leaky_relu', initialization='he', dropout_rate=0.2),
        LayerConfig(size=32, activation='leaky_relu', initialization='he', dropout_rate=0.2),
        LayerConfig(size=16, activation='leaky_relu', initialization='he', dropout_rate=0.1),
        LayerConfig(size=2, activation='sigmoid', initialization='xavier')
    ]
    
    # Try multiple random initializations
    best_accuracy = 0
    best_model = None
    
    print("Training models with different initializations...")
    for attempt in range(3):
        np.random.seed()  # Reset random seed
        
        model = NeuralNetwork(
            layer_configs=layers,
            optimizer='rmsprop',
            learning_rate=0.01,  # Lower learning rate for stability
            momentum_beta=0.9,
            batch_norm=True,
            gradient_clip=1.0
        )
        
        # Train with mini-batches
        model.train(X_train, y_train, epochs=1000, batch_size=32)
        
        # Evaluate
        predictions = model.predict(X_test)
        binary_predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(np.all(binary_predictions == y_test, axis=0))
        
        print(f"Attempt {attempt + 1} accuracy: {accuracy:.2%}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    print(f"\nBest model accuracy: {best_accuracy:.2%}")
    
    # Visualize results with best model
    plt.figure(figsize=(15, 5))
    
    # Plot training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[0, y_train[0] == 1], X_train[1, y_train[0] == 1], 
               c='blue', marker='o', label='Class 1 (Train)', alpha=0.6)
    plt.scatter(X_train[0, y_train[1] == 1], X_train[1, y_train[1] == 1], 
               c='red', marker='o', label='Class 2 (Train)', alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    
    # Plot test data with predictions
    plt.subplot(1, 2, 2)
    predictions = best_model.predict(X_test)
    binary_predictions = (predictions > 0.5).astype(int)
    correct_mask = np.all(binary_predictions == y_test, axis=0)
    incorrect_mask = ~correct_mask
    
    # Plot correct predictions
    plt.scatter(X_test[0, correct_mask & (y_test[0] == 1)], 
               X_test[1, correct_mask & (y_test[0] == 1)],
               c='blue', marker='o', label='Class 1 (Correct)', alpha=0.6)
    plt.scatter(X_test[0, correct_mask & (y_test[1] == 1)], 
               X_test[1, correct_mask & (y_test[1] == 1)],
               c='red', marker='o', label='Class 2 (Correct)', alpha=0.6)
    
    # Plot incorrect predictions
    plt.scatter(X_test[0, incorrect_mask], X_test[1, incorrect_mask],
               c='black', marker='x', label='Incorrect', alpha=0.6, s=100)
    
    plt.title('Test Data Predictions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    
    # Plot decision boundary
    x_min, x_max = X_test[0].min() - 0.5, X_test[0].max() + 0.5
    y_min, y_max = X_test[1].min() - 0.5, X_test[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = np.array([xx.ravel(), yy.ravel()])
    
    Z = best_model.predict(grid_points)
    Z = Z[0].reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing Neural Network on Unseen Data")
    print("=====================================")
    
    test_xor_generalization()
    test_spiral_classification()
