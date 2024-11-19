import numpy as np
from lib.lousybook01.LousyBookML import NeuralNetwork, LayerConfig

def create_xor_dataset():
    """Create the XOR dataset for binary classification."""
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    return X, y

def train_and_evaluate(nn, X, y, name="", learning_rate=0.1):
    """Train a neural network and evaluate its performance."""
    print(f"\nTraining neural network with {name}:")
    nn.train(X, y, learning_rate=learning_rate, epochs=1000, batch_size=4)
    
    predictions = nn.predict(X)
    print("\nFinal Predictions:")
    print("Input -> Output (Expected)")
    for i in range(4):
        print(f"[{X[0,i]}, {X[1,i]}] -> {float(predictions[0,i]):.4f} ({int(y[0,i])})")

def main():
    # Create dataset
    X, y = create_xor_dataset()

    # Test different network architectures with customized layers
    architectures = [
        {
            "name": "Mixed Activation Network",
            "layers": [
                LayerConfig(size=2),  # Input layer
                LayerConfig(size=4, activation='relu', initialization='he',
                          dropout_rate=0.1, l2_reg=0.01),
                LayerConfig(size=4, activation='leaky_relu', initialization='he',
                          dropout_rate=0.1, l2_reg=0.01),
                LayerConfig(size=1, activation='sigmoid', initialization='xavier')
            ],
            "params": {
                "optimizer": "momentum",
                "momentum_beta": 0.9,
                "batch_norm": True,
                "gradient_clip": 1.0,
                "epsilon": 1e-5
            },
            "learning_rate": 0.1
        },
        {
            "name": "Deep ReLU Network",
            "layers": [
                LayerConfig(size=2),  # Input layer
                LayerConfig(size=8, activation='relu', initialization='orthogonal',
                          dropout_rate=0.2, l2_reg=0.01),
                LayerConfig(size=4, activation='relu', initialization='orthogonal',
                          dropout_rate=0.1, l2_reg=0.01),
                LayerConfig(size=1, activation='sigmoid', initialization='xavier')
            ],
            "params": {
                "optimizer": "rmsprop",
                "rmsprop_beta": 0.999,
                "batch_norm": True,
                "gradient_clip": 1.0,
                "epsilon": 1e-5
            },
            "learning_rate": 0.03
        },
        {
            "name": "Hybrid Network",
            "layers": [
                LayerConfig(size=2),  # Input layer
                LayerConfig(size=4, activation='tanh', initialization='xavier',
                          dropout_rate=0.1),
                LayerConfig(size=4, activation='relu', initialization='he',
                          dropout_rate=0.1),
                LayerConfig(size=1, activation='sigmoid', initialization='xavier')
            ],
            "params": {
                "optimizer": "momentum",
                "momentum_beta": 0.9,
                "batch_norm": False,
                "gradient_clip": 1.0,
                "epsilon": 1e-5
            },
            "learning_rate": 0.08
        }
    ]

    # Train and evaluate each architecture
    for arch in architectures:
        nn = NeuralNetwork(
            layer_configs=arch["layers"],
            learning_rate=arch["learning_rate"],
            **arch["params"]
        )
        train_and_evaluate(nn, X, y, arch["name"], arch["learning_rate"])

if __name__ == "__main__":
    main()