import numpy as np
import matplotlib.pyplot as plt
from lousybook01.LousyBookML import NeuralNetwork, LayerConfig

def xor_example():
    """Demonstrate XOR problem solution with different architectures."""
    # XOR data
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Different architectures to try
    architectures = [
        {
            "name": "Simple ReLU Network",
            "layers": [
                LayerConfig(size=2),
                LayerConfig(size=4, activation='relu'),
                LayerConfig(size=1, activation='sigmoid')
            ],
            "optimizer": "momentum"
        },
        {
            "name": "Deep ReLU Network",
            "layers": [
                LayerConfig(size=2),
                LayerConfig(size=8, activation='relu'),
                LayerConfig(size=4, activation='relu'),
                LayerConfig(size=1, activation='sigmoid')
            ],
            "optimizer": "rmsprop"
        },
        {
            "name": "Mixed Activation Network",
            "layers": [
                LayerConfig(size=2),
                LayerConfig(size=4, activation='tanh'),
                LayerConfig(size=4, activation='relu'),
                LayerConfig(size=1, activation='sigmoid')
            ],
            "optimizer": "momentum"
        }
    ]
    
    results = {}
    for arch in architectures:
        print(f"\nTraining {arch['name']}...")
        model = NeuralNetwork(
            layer_configs=arch["layers"],
            optimizer=arch["optimizer"],
            learning_rate=0.1
        )
        
        # Train and collect metrics
        model.train(X, y, epochs=1000, batch_size=4)
        predictions = model.predict(X)
        
        results[arch["name"]] = {
            "predictions": predictions,
            "active_neurons": model.active_neurons[-100:],  # Last 100 epochs
            "gradient_norms": model.gradient_norms[-100:]
        }
        
        print(f"Final predictions for {arch['name']}:")
        print("Input -> Output (Expected)")
        for i in range(4):
            print(f"[{X[0,i]}, {X[1,i]}] -> {predictions[0,i]:.4f} ({y[0,i]})")
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data["active_neurons"], label=name)
    plt.title("Active Neurons Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("% Active Neurons")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["gradient_norms"], label=name)
    plt.title("Gradient Norms Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Gradient Norm")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def classification_example():
    """Demonstrate binary classification on generated data."""
    # Generate spiral data
    def generate_spiral_data(points_per_class, classes):
        X = []
        y = []
        for class_idx in range(classes):
            r = np.linspace(0.0, 1, points_per_class)
            t = np.linspace(class_idx * 4, (class_idx + 1) * 4, points_per_class) + \
                np.random.randn(points_per_class) * 0.2
            X.append(np.column_stack((r * np.sin(t * 2.5), r * np.cos(t * 2.5))))
            y.append(np.zeros((points_per_class, classes)))
            y[-1][:, class_idx] = 1
        return np.vstack(X).T, np.vstack(y).T

    # Generate data
    X, y = generate_spiral_data(100, 2)
    
    # Create model
    layers = [
        LayerConfig(size=2),
        LayerConfig(size=16, activation='relu', dropout_rate=0.1),
        LayerConfig(size=8, activation='relu', dropout_rate=0.1),
        LayerConfig(size=2, activation='sigmoid')
    ]
    
    model = NeuralNetwork(
        layer_configs=layers,
        optimizer='momentum',
        learning_rate=0.05,
        batch_norm=True
    )
    
    # Train
    print("\nTraining classification model...")
    model.train(X, y, epochs=1000, batch_size=32)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 10))
    
    # Generate grid of points
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Predict for grid points
    grid_points = np.array([xx.ravel(), yy.ravel()])
    Z = model.predict(grid_points)
    Z = Z[0].reshape(xx.shape)
    
    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[0], X[1], c=y[0], cmap=plt.cm.RdYlBu)
    plt.title("Neural Network Decision Boundary")
    plt.show()

if __name__ == "__main__":
    print("Running Neural Network Examples...")
    
    xor_example()
    classification_example()
    
    print("\nAll examples completed!")
