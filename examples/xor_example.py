from LousyBookML import NeuralNetwork
from LousyBookML.neural_network.activations import relu, sigmoid, tanh, softmax, leaky_relu
from LousyBookML.neural_network.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
from LousyBookML.neural_network.model import Layer
from LousyBookML.neural_network.utils import normalize_data, to_categorical
import numpy as np

model = NeuralNetwork([
        Layer(units=16, activation='relu'),
        Layer(units=1, activation='sigmoid')
    ], loss='mean_squared_error', optimizer='adam', clip_value=10)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Generate random test data
numberOfPredictionData = np.random.randint(10, 50)  # Random number between 10 and 50
unseenX = np.random.randint(0, 2, size=(numberOfPredictionData, 2))  # Random binary pairs
resultX = np.array([int(x[0] ^ x[1]) for x in unseenX])  # XOR operation using ^

history = model.fit(X, Y, epochs=200, verbose=True)

prediction = model.predict(unseenX)
print(f"\nTesting model with {numberOfPredictionData} random samples:")
print("Random test data shape:", unseenX.shape)
print("\nPredictions:")
correct_predictions = 0
for i in range(numberOfPredictionData):
    rounded_pred = int(np.round(prediction[i][0]))
    is_correct = rounded_pred == resultX[i]
    if is_correct:
        correct_predictions += 1
    print(f"For {unseenX[i]}, XOR = {resultX[i]}, Model Predicted {prediction[i][0]:.4f}, Rounded to {rounded_pred}, Correct: {is_correct}")

accuracy = (correct_predictions / numberOfPredictionData) * 100
print(f"\nAccuracy on random test data: {accuracy:.2f}%")