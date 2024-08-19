import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QHBoxLayout
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import Qt

# Neural Network with Enhanced Memory
class RockPaperScissorsAI:
    def __init__(self, learning_rate=0.1):
        self.input_size = 3
        self.hidden_size = 10
        self.output_size = 3
        self.learning_rate = learning_rate

        # Initialize weights
        self.W_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.W_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.W_hidden_hidden = np.random.randn(self.hidden_size, self.hidden_size) * 0.01

        self.hidden_memory = np.zeros((1, self.hidden_size))

        # Memory to store past moves
        self.memory = []

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_memory = np.tanh(np.dot(x, self.W_input_hidden) + np.dot(self.hidden_memory, self.W_hidden_hidden))
        output_layer = self.softmax(np.dot(self.hidden_memory, self.W_hidden_output))
        return output_layer

    def train(self, x, y):
        # Forward pass
        output = self.forward(x)

        # Calculate loss (cross-entropy)
        error_output = output - y
        error_hidden = (1 - self.hidden_memory ** 2) * np.dot(error_output, self.W_hidden_output.T)

        # Backpropagation
        self.W_hidden_output -= self.learning_rate * np.dot(self.hidden_memory.T, error_output)
        self.W_input_hidden -= self.learning_rate * np.dot(x.T, error_hidden)
        self.W_hidden_hidden -= self.learning_rate * np.dot(self.hidden_memory.T, error_hidden)

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output)

    def move_to_vector(self, move):
        return {'rock': [1, 0, 0], 'paper': [0, 1, 0], 'scissors': [0, 0, 1]}[move]

    def vector_to_move(self, vector):
        return ['rock', 'paper', 'scissors'][vector]

    def update_memory(self, player_move, ai_move):
        self.memory.append((self.move_to_vector(player_move), self.move_to_vector(ai_move)))
        if len(self.memory) > 10:  # Keep a memory of the last 10 rounds
            self.memory.pop(0)

    def learn_from_memory(self):
        for player_vector, ai_vector in self.memory:
            self.train(np.array(player_vector).reshape(1, -1), np.array(ai_vector).reshape(1, -1))

# PyQt6 GUI for Rock-Paper-Scissors Game
class RPSGame(QWidget):
    def __init__(self):
        super().__init__()
        self.ai = RockPaperScissorsAI()
        self.initUI()
        self.init_training_data()
        self.last_player_move = None
        self.last_ai_move = None

    def initUI(self):
        self.setWindowTitle('Rock Paper Scissors AI')
        self.setGeometry(100, 100, 400, 300)

        # Modern color palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.Text, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(50, 50, 50))
        self.setPalette(palette)

        # Layout and widgets
        layout = QVBoxLayout()

        self.result_label = QLabel('Welcome to Rock-Paper-Scissors!')
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        # Learning Rate Slider
        slider_layout = QHBoxLayout()
        slider_label = QLabel('Learning Rate:')
        self.learning_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.learning_rate_slider.setMinimum(1)
        self.learning_rate_slider.setMaximum(1000)
        self.learning_rate_slider.setValue(100)  # Default value (0.1)
        self.learning_rate_slider.setTickInterval(1)
        self.learning_rate_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.learning_rate_slider.valueChanged.connect(self.update_learning_rate)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.learning_rate_slider)
        layout.addLayout(slider_layout)

        # Buttons for Rock, Paper, Scissors
        self.buttons = {
            'rock': QPushButton('Rock'),
            'paper': QPushButton('Paper'),
            'scissors': QPushButton('Scissors')
        }

        for key, button in self.buttons.items():
            button.setStyleSheet("QPushButton { background-color: #323232; color: #F0F0F0; padding: 10px; border-radius: 5px; }")
            button.clicked.connect(lambda _, move=key: self.play(move))
            layout.addWidget(button)

        self.setLayout(layout)

    def update_learning_rate(self):
        lr_value = self.learning_rate_slider.value() / 1000.0
        self.ai.set_learning_rate(lr_value)
        self.result_label.setText(f'Learning Rate: {lr_value:.3f}')

    def init_training_data(self):
        # Generate random training data
        moves = ['rock', 'paper', 'scissors']
        training_data = [np.random.choice(moves) for _ in range(100)]
        X_train = np.array([self.ai.move_to_vector(move) for move in training_data])
        Y_train = np.roll(X_train, -1, axis=0)

        # Train the AI with random data
        for x, y in zip(X_train, Y_train):
            self.ai.train(x.reshape(1, -1), y.reshape(1, -1))

    def play(self, player_move):
        # AI makes its prediction
        player_vector = self.ai.move_to_vector(player_move)
        ai_move_index = self.ai.predict(player_vector)
        ai_predicted_move = self.ai.vector_to_move(ai_move_index)
        result = self.determine_winner(player_move, ai_predicted_move)

        # Update UI
        self.result_label.setText(f'You: {player_move}, AI: {ai_predicted_move}, {result}')

        # Train AI after the round is completed
        if self.last_player_move is not None and self.last_ai_move is not None:
            self.ai.update_memory(self.last_player_move, self.last_ai_move)
            self.ai.learn_from_memory()

        # Store current moves for next round's training
        self.last_player_move = player_move
        self.last_ai_move = ai_predicted_move

    def determine_winner(self, player_move, ai_move):
        if player_move == ai_move:
            return "It's a tie!"
        elif (player_move == 'rock' and ai_move == 'scissors') or \
             (player_move == 'paper' and ai_move == 'rock') or \
             (player_move == 'scissors' and ai_move == 'paper'):
            return "You win!"
        else:
            return "AI wins!"

# Main application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = RPSGame()
    game.show()
    sys.exit(app.exec())
