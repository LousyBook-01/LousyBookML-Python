# LousyBookML

This repository contains a simple machine learning library written in Python. It currently features a neural network implementation (in `lib/lousybook01/LousyBookML.py`) and is designed to be easy to use.

## Features

- **Neural Network:**  The core of the library is a feedforward neural network with configurable architecture, including the number of input, hidden, and output neurons.
- **Training:** The network can be trained using backpropagation with RMSprop optimization for improved training stability.
- **Activation Functions:**  Currently uses Leaky ReLU for hidden layers and Sigmoid for the output layer.
- **Saving and Loading:**  Models can be saved and loaded for later use.

## Getting Started

1. **Install Dependencies:**
    `
    pip install -r requirements.txt
    `

2. **Run the Example:**
    `
    python main.py
    `
    The example demonstrates how to create a neural network, train it on XOR data, and then make predictions. To use this in your own project, copy the lib folder to your project folder, then in the script you want to use this in add `import lib.lousybook01.LousyBookML`

## TODO

- **Add More Machine Learning Algorithms:**
    Expand the library to include other popular algorithms like support vector machines, decision trees, and more.
- **Documentation:**
    Create detailed documentation for all classes, methods, and features.
- **Fix some performance issues:**
    Explore ways to optimize the code for better performance and speed.
- **Error Handling:**
    Currently I have no error handling system for this. I should add more robust error handling and validation checks

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## FAQ

- **When will I update it?**
    Probably when I have some free time, bcuz I have school, exams and homework.
- **Will it be more advanced in the future and have more features?**
    Probably I guess idk, depends on my motivation and the attention this gets. And also my brain lol.

Made By LousyBook01 2024, professional idiot
