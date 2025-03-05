import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from PIL import Image, ImageDraw


mnist = fetch_openml('mnist_784', version=1, parser="auto")
x, y = mnist['data'], mnist['target']

y = y.astype(int)

print("Shape of X:", x.shape) 


ch = np.array(x.iloc[30080]) 
ch_img = ch.reshape(28, 28) 

print("Label:", y.iloc[30080])
# plt.imshow(ch_img, cmap='gray', interpolation='nearest')
# plt.show()
# h

x_train, x_test = x.iloc[:60000], x.iloc[60000:]
y_train, y_test = y.iloc[:60000], y.iloc[60000:]


lr = LogisticRegression(max_iter=25, solver='saga', n_jobs=-1)
lr.fit(x_train, y_train)

predictions = lr.predict(x_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")

# #
# Function to create canvas input with a redraw option
def draw_digit():
    """Opens a Tkinter canvas for the user to draw a digit and returns a 1D NumPy array (28x28)."""
    root = tk.Tk()
    root.title("Draw a Digit")

    canvas_size = 280  # 10x the MNIST image size for better drawing
    image_size = 28  # Final image size for the model
    bg_color = "black"
    fg_color = "white"

    # Create a blank image with PIL
    image = Image.new("L", (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(image)

    # Function to draw on the canvas
    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill=fg_color, outline=fg_color)
        draw.ellipse([x1, y1, x2, y2], fill=fg_color, outline=fg_color)

    # Function to clear the canvas
    def clear_canvas():
        canvas.delete("all")
        draw.rectangle([0, 0, canvas_size, canvas_size], fill=bg_color)

    # Function to save the drawn digit and close the window
    def save_and_close():
        nonlocal image
        image = image.resize((image_size, image_size))  # Resize to 28x28
        image = np.array(image)  # Convert to NumPy array
        image = image / 255.0  # Normalize pixel values
        image = 1 - image  # Invert colors (since MNIST digits are black on white)
        image = image.flatten()  # Flatten to (784,) to match MNIST format
        root.quit()
        root.destroy()

    # Create the canvas
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg=bg_color)
    canvas.pack()

    # Bind mouse events for drawing
    canvas.bind("<B1-Motion>", paint)

    # Buttons for clear and submit
    button_frame = tk.Frame(root)
    button_frame.pack()

    clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
    clear_button.pack(side=tk.LEFT, padx=10)

    submit_button = tk.Button(button_frame, text="Submit", command=save_and_close)
    submit_button.pack(side=tk.RIGHT, padx=10)

    root.mainloop()  # Run the Tkinter event loop
    
    return image  # Return processed 28x28 image array


# Get user-drawn digit as an array
digit_array = draw_digit()

# Predict using the trained logistic regression model
prediction = lr.predict([digit_array])
print(f"Predicted Digit: {prediction[0]}")
