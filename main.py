import numpy as np
import random
import lib.lousybook01.LousyBookML as LousyBookML # or just "import lib" or "import lib.lousybook" but then you need to change the neural network creation and stuff

# Example usage:
if __name__ == "__main__":
    # Define input, hidden, and output sizes (with default values)
    # input_size = 2
    # hidden_size = 3
    # output_size = 1
    # 
    # # Create a neural network instance
    # nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn = LousyBookML.NeuralNetwork()
    # Define training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the network
    nn.train(X, y, debug=True)

    # # Save the trained model
    # nn.save_model('my_trained_model.pkl')

    # Load the saved model 
    # loaded_model = NeuralNetwork.load_model('my_trained_model.pkl')

    # Make predictions using the current neural network (nn)
    for _ in range(5):  # Make 5 predictions
        # Generate random input data
        random_input = [random.randint(0, 1) for _ in range(2)]
        random_input = np.array(random_input).reshape(1, -1)  # Reshape for prediction

        # Make prediction with the random input
        prediction = nn.predict(random_input, debug=False)
        print(f"Input: {random_input}, Prediction: {prediction[0][0]}")