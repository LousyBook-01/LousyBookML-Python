import time
print("Importing Modules...")

start = time.time()
import numpy as np
import gzip
import os
import urllib.request
from LousyBookML.neural_network.model import NeuralNetwork, Layer
from LousyBookML.neural_network.optimizers import SGD, Adam
from LousyBookML.scalers import StandardScaler
import matplotlib.pyplot as plt
import pickle

print("Loaded modules in ", time.time() - start)

def fetch_mnist():
    """Download and load MNIST dataset from mirror"""
    def download(filename, source='https://storage.googleapis.com/cvdf-datasets/mnist/'):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(source + filename, filename)

    def load_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    # Check if cached data exists
    cache_file = 'examples/mnist_data.pkl'
    if os.path.exists(cache_file):
        print("Loading MNIST from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Downloading MNIST dataset...")
    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test = load_images('t10k-images-idx3-ubyte.gz')
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')

    # Save to cache
    print("Saving MNIST to cache...")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)

    # Clean up downloaded files
    for fname in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        if os.path.exists(fname):
            os.remove(fname)

    return X_train, y_train, X_test, y_test

# Load MNIST dataset
print("Loading MNIST dataset...")
X_train, y_train, X_test, y_test = fetch_mnist()

# One-hot encode the target
y_train_onehot = np.zeros((y_train.size, 10))
y_train_onehot[np.arange(y_train.size), y_train] = 1
y_test_onehot = np.zeros((y_test.size, 10))
y_test_onehot[np.arange(y_test.size), y_test] = 1

# Create data augmentation function
def augment_with_straight_digits(X, y, y_onehot, num_variations=2, chunk_size=1000):
    """Augment dataset with straight-line variations using chunked processing."""
    print("Augmenting dataset with straight-line variations in chunks...")
    n_samples = X.shape[0]
    total_samples = n_samples * (num_variations + 1)
    
    # Initialize arrays for final results
    augmented_X = np.zeros((total_samples, 784), dtype=X.dtype)
    augmented_y = np.zeros(total_samples, dtype=y.dtype)
    augmented_y_onehot = np.zeros((total_samples, y_onehot.shape[1]), dtype=y_onehot.dtype)
    
    # Process original data in chunks
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunk_size_actual = end_idx - i
        
        # Copy original chunk data
        augmented_X[i:end_idx] = X[i:end_idx].reshape(chunk_size_actual, -1)
        augmented_y[i:end_idx] = y[i:end_idx]
        augmented_y_onehot[i:end_idx] = y_onehot[i:end_idx]
        
        # Create variations for this chunk
        for v in range(num_variations):
            start_aug_idx = n_samples * (v + 1) + i
            end_aug_idx = n_samples * (v + 1) + end_idx
            
            # Create thresholded version for this chunk
            chunk_imgs = X[i:end_idx].copy()
            chunk_imgs = np.where(chunk_imgs > 127, 255, 0)
            
            # Store augmented chunk
            augmented_X[start_aug_idx:end_aug_idx] = chunk_imgs.reshape(chunk_size_actual, -1)
            augmented_y[start_aug_idx:end_aug_idx] = y[i:end_idx]
            augmented_y_onehot[start_aug_idx:end_aug_idx] = y_onehot[i:end_idx]
        
        if (i + chunk_size) % 5000 == 0:
            print(f"Processed {i + chunk_size}/{n_samples} samples...")
    
    print("Augmentation complete!")
    return augmented_X, augmented_y, augmented_y_onehot

# Prepare and scale the data
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
X_test_reshaped = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

# Fit and transform training data
X_train_norm = scaler.fit_transform(X_train_reshaped)
# Transform test data using same scaler
X_test_norm = scaler.transform(X_test_reshaped)

# Optional: Augment training data (comment out if not needed)
# X_train_aug, y_train_aug, y_train_onehot_aug = augment_with_straight_digits(
#     X_train, y_train, y_train_onehot, num_variations=1)
# X_train_aug_norm = scaler.transform(X_train_aug.astype('float32') / 255.0)

# Use augmented or original data
use_augmentation = False  # Enable augmentation
if use_augmentation:
    X_train_aug, y_train_aug, y_train_onehot_aug = augment_with_straight_digits(
        X_train, y_train, y_train_onehot, num_variations=5, chunk_size=1000)  # Process in chunks of 1000
    X_train_aug_norm = scaler.transform(X_train_aug.astype('float32') / 255.0)
    train_X = X_train_aug_norm
    train_y = y_train_onehot_aug
else:
    train_X = X_train_norm
    train_y = y_train_onehot

# Create and train the model with optimized architecture
model = NeuralNetwork([
    {'units': 64, 'activation': 'relu', 'batch_norm': False},    # Larger first layer with batch norm
    {'units': 32, 'activation': 'relu', 'batch_norm': False},    # Second hidden layer
    {'units': 10, 'activation': 'softmax'}                       # Output layer with softmax
], loss='mse',  # Better for classification
   optimizer='adam', 
   learning_rate=0.001)  # Good learning rate with batch norm

# Initialize the model with the correct input size (28x28 = 784)
model.initialize(784)

def calculate_accuracy(model, X, y_true):
    """Calculate accuracy for the given data."""
    predictions = model.predict(X)
    if len(y_true.shape) > 1:  # If one-hot encoded
        y_true = np.argmax(y_true, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == y_true)

# Train the model with early stopping and print accuracy
print("Training model...")
for epoch in range(20):
    # Train for one epoch
    history = model.fit(train_X, train_y, 
                       epochs=1,
                       batch_size=128,           
                       validation_data=(X_test_norm, y_test_onehot),
                       early_stopping_patience=5,
                       early_stopping_min_delta=1e-4,
                       num_verbose_prints=1)
    
    # Calculate and print accuracies
    train_acc = calculate_accuracy(model, train_X, train_y)
    val_acc = calculate_accuracy(model, X_test_norm, y_test)
    print(f"Epoch {epoch+1}/20:")
    print(f"  Training Loss: {history['loss'][-1]:.4f}")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")

# Print final metrics
final_train_acc = calculate_accuracy(model, train_X, train_y)
final_val_acc = calculate_accuracy(model, X_test_norm, y_test)
print("\nFinal Results:")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Save the trained weights
print("Saving model weights and scaler...")
weights = []
for layer in model.layers:
    weights.append({
        'weights': layer.weights,
        'bias': layer.bias
    })
with open('examples/model_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)

# Save the scaler
with open('examples/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 1, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Evaluate on test set
print("\nEvaluating model on test set...")
test_pred = model.predict(X_test_norm)
test_accuracy = np.mean(np.argmax(test_pred, axis=1) == y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save some example predictions
n_examples = 10
fig, axes = plt.subplots(2, n_examples, figsize=(20, 4))
for i in range(n_examples):
    # Original digit
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    pred = np.argmax(test_pred[i])
    true = y_test[i]
    axes[0, i].set_title(f'Pred: {pred}\nTrue: {true}')
    
    # Add some grid lines to help visualize straight lines
    axes[1, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[1, i].grid(True, color='red', alpha=0.3)
    axes[1, i].axis('off')

plt.savefig('examples/predictions.png')
plt.close()

print("Training complete! Check examples/training_history.png and examples/predictions.png for visualizations.")