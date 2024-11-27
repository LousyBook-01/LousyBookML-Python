import pygame
import numpy as np
from LousyBookML.neural_network.model import NeuralNetwork, Layer
from LousyBookML.neural_network.optimizers import SGD
from LousyBookML.scalers import StandardScaler
import sys
import pickle

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 670  # Increased from 600 to accommodate network input preview
GRID_SIZE = 280
PADDING = 15
SIDEBAR_WIDTH = 200
INPUT_PREVIEW_SIZE = 150  # Size for the network input preview
BRUSH_SIZE = 10  # Thinner brush size
CELL_SIZE = 10  # Size of each cell in the preview grid

# Colors (Dark theme)
DARK_BG = (30, 30, 30)
DARKER_BG = (20, 20, 20)
LIGHT_GRAY = (200, 200, 200)
BORDER_COLOR = (60, 60, 60)
ACCENT_COLOR = (0, 255, 255)
HIGHLIGHT_COLOR = (0, 100, 100)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Create window with space for preview and confidence bars
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Digit Recognition (Press 'R' to reset)")

# Initialize drawing surface with dark background
drawing_surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
drawing_surface.fill(WHITE)  # Keep white for drawing area

def draw_bordered_rect(surface, color, rect, border_color=BORDER_COLOR, border_width=2):
    """Draw a rectangle with a border."""
    pygame.draw.rect(surface, border_color, rect)
    inner_rect = pygame.Rect(rect.left + border_width, rect.top + border_width,
                           rect.width - 2*border_width, rect.height - 2*border_width)
    pygame.draw.rect(surface, color, inner_rect)

def draw_confidence_bar(y_pos, confidence, digit):
    """Draw a confidence bar with percentage for a digit."""
    # Draw digit label
    font = pygame.font.Font(None, 24)
    text = font.render(f"{digit}:", True, LIGHT_GRAY)
    screen.blit(text, (GRID_SIZE + PADDING * 3, y_pos))
    
    # Draw bar background
    bar_x = GRID_SIZE + PADDING * 4 + 20
    bar_width = SIDEBAR_WIDTH - PADDING * 6 - 70  # Leave space for percentage
    pygame.draw.rect(screen, DARKER_BG, (bar_x, y_pos, bar_width, 15))
    
    # Draw filled portion of bar
    fill_width = int(bar_width * confidence)
    if fill_width > 0:
        pygame.draw.rect(screen, ACCENT_COLOR, (bar_x, y_pos, fill_width, 15))
    
    # Draw percentage text
    percentage = f"{confidence*100:.1f}%"
    text = font.render(percentage, True, LIGHT_GRAY)
    screen.blit(text, (bar_x + bar_width + 10, y_pos))

def draw_prediction(prediction, input_image):
    """Draw the model's prediction and confidence bars."""
    # Convert outputs to probabilities
    shifted = prediction - np.min(prediction)
    if np.max(shifted) > 0:
        shifted = shifted / np.max(shifted)
    exp_preds = np.exp(shifted * 5)
    probabilities = exp_preds / exp_preds.sum()
    predicted_digit = np.argmax(probabilities)
    
    # Calculate total height needed for predictions
    font = pygame.font.Font(None, 32)
    title_text = font.render("Predictions", True, LIGHT_GRAY)
    title_height = title_text.get_height()
    
    # Height calculation for all elements
    total_content_height = (title_height + PADDING * 3 + # Title and padding
                          10 * 35 +  # Height for 10 confidence bars
                          PADDING * 2)  # Bottom padding
    
    # Clear and draw sidebar background
    sidebar_rect = pygame.Rect(GRID_SIZE + PADDING, PADDING, 
                             SIDEBAR_WIDTH - PADDING * 2, total_content_height)
    draw_bordered_rect(screen, DARK_BG, sidebar_rect)
    
    # Draw title
    screen.blit(title_text, (GRID_SIZE + PADDING * 2, PADDING * 2))
    
    # Draw confidence bars
    for i in range(10):
        y = PADDING * 3 + title_height + i * 35
        if i == predicted_digit:
            highlight_rect = pygame.Rect(GRID_SIZE + PADDING * 2, y - 2, 
                                       SIDEBAR_WIDTH - PADDING * 4, 20)
            pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)
        draw_confidence_bar(y, probabilities[i], i)
    
    # Draw network input visualization on the right side
    preview_x = GRID_SIZE + SIDEBAR_WIDTH + PADDING
    preview_y = PADDING
    
    # Draw title for network input
    input_title = font.render("Network Input", True, LIGHT_GRAY)
    screen.blit(input_title, (preview_x, preview_y))
    
    # Create preview area
    input_rect = pygame.Rect(preview_x, preview_y + input_title.get_height() + PADDING,
                           INPUT_PREVIEW_SIZE, INPUT_PREVIEW_SIZE)
    draw_bordered_rect(screen, DARK_BG, input_rect)
    
    preview_surface = pygame.Surface((input_rect.width, input_rect.height))
    preview_surface.fill(DARK_BG)
    
    # Draw grid lines
    cell_size = input_rect.width / 28
    for i in range(29):
        pos = i * cell_size
        pygame.draw.line(preview_surface, BORDER_COLOR, (pos, 0), (pos, input_rect.height), 1)
        pygame.draw.line(preview_surface, BORDER_COLOR, (0, pos), (input_rect.width, pos), 1)
    
    # Draw actual pixels
    for i in range(28):
        for j in range(28):
            val = min(255, max(0, int(255 - input_image[i, j] * 255)))
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            pygame.draw.rect(preview_surface, (val, val, val), rect)
    
    # Draw the preview
    screen.blit(preview_surface, input_rect)
    pygame.display.flip()

def reset_canvas():
    """Reset the drawing surface."""
    global drawing_surface
    screen.fill(DARK_BG)
    drawing_surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
    drawing_surface.fill(WHITE)
    # Draw border around drawing area
    draw_bordered_rect(screen, DARK_BG, pygame.Rect(0, 0, GRID_SIZE, GRID_SIZE))
    screen.blit(drawing_surface, (0, 0))
    pygame.display.flip()

def draw_brush(pos, size=BRUSH_SIZE, last_pos=None):
    """Draw a smooth brush stroke."""
    if last_pos is None:
        # Just draw a circle if no last position
        pygame.draw.circle(drawing_surface, BLACK, pos, size//2)
    else:
        # Draw a line between last_pos and current pos for smooth stroke
        pygame.draw.line(drawing_surface, BLACK, last_pos, pos, size)
        # Add circle at current pos to smooth out the line ends
        pygame.draw.circle(drawing_surface, BLACK, pos, size//2)
    
    # Update the screen with the new drawing
    screen.blit(drawing_surface, (0, 0))
    pygame.display.update()

def process_image():
    """Process the drawing surface for prediction."""
    # Get pixel array from drawing surface and transpose to correct orientation
    pixel_array = pygame.surfarray.array2d(drawing_surface).T
    
    # Invert colors since we draw in black on white
    pixel_array = np.where(pixel_array == 0, 255, 0)
    
    # Create a larger numpy array for visualization
    visual_array = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Resize to 28x28 with better handling of straight lines
    scaled = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            y_start = int(i * GRID_SIZE / 28)
            x_start = int(j * GRID_SIZE / 28)
            y_end = int((i + 1) * GRID_SIZE / 28)
            x_end = int((j + 1) * GRID_SIZE / 28)
            
            # Take maximum value in the region to preserve thin lines
            region = pixel_array[y_start:y_end, x_start:x_end]
            scaled[i, j] = np.max(region)
            
            # Fill the visualization array
            visual_array[y_start:y_end, x_start:x_end] = scaled[i, j]
    
    # Normalize to [0, 1] range
    scaled = scaled / 255.0
    visual_array = visual_array / 255.0
    
    # Draw the visual array to show what's being fed to the network
    preview_surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
    preview_surface.fill(DARK_BG)
    
    # Draw grid lines
    for i in range(29):
        pos = i * (GRID_SIZE // 28)
        pygame.draw.line(preview_surface, BORDER_COLOR, (pos, 0), (pos, GRID_SIZE), 1)
        pygame.draw.line(preview_surface, BORDER_COLOR, (0, pos), (GRID_SIZE, pos), 1)
    
    # Draw actual pixels with transposed coordinates
    cell_size = GRID_SIZE // 28
    for i in range(28):
        for j in range(28):
            val = min(255, max(0, int(255 - scaled[i, j] * 255)))
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            pygame.draw.rect(preview_surface, (val, val, val), rect)
    
    # Draw title for input preview
    font = pygame.font.Font(None, 32)
    text = font.render("Network Input (28x28)", True, LIGHT_GRAY)
    screen.blit(text, (PADDING, GRID_SIZE + PADDING * 2))
    
    # Draw the preview with grid
    screen.blit(preview_surface, (0, GRID_SIZE + PADDING))
    pygame.display.flip()
    
    # Normalize and reshape for the neural network
    scaled = scaled.reshape(1, -1)
    scaled = scaler.transform(scaled)
    
    # Make prediction
    prediction = model.predict(scaled)
    draw_prediction(prediction[0], scaled.reshape(28, 28))

# Create the model with the same architecture as training
model = NeuralNetwork([
    {'units': 128, 'activation': 'relu'},    # Single hidden layer
    {'units': 10, 'activation': 'linear'}    # Output layer
], loss='mse', optimizer='adam')

# Initialize the model
model.initialize(784)  # 28x28 = 784 input features

try:
    with open('examples/model_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
        for layer, w in zip(model.layers, weights):
            layer.weights = w['weights']
            layer.bias = w['bias']
    # Load the scaler that was used during training
    with open('examples/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Successfully loaded trained weights and scaler!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please train the model first using nn_mnist_example.py")
    sys.exit(1)

# Main loop
drawing = False
last_pos = None
reset_canvas()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[0] < GRID_SIZE:  # Only draw in the drawing area
                drawing = True
                pos = event.pos
                draw_brush(pos)
                last_pos = pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
            process_image()
        elif event.type == pygame.MOUSEMOTION:
            if drawing and event.pos[0] < GRID_SIZE:
                pos = event.pos
                draw_brush(pos, last_pos=last_pos)
                last_pos = pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Reset canvas when 'R' is pressed
                reset_canvas()
                last_pos = None

pygame.quit()
