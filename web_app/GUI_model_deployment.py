import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from PIL import Image, ImageTk
from keras.models import load_model
import pickle
from keras_preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('final_model_load/final_sentiment_model.h5')

# Load the tokenizer
with open('final_model_load/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the label encoder
with open('final_model_load/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def predict(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=256)
    pred = model.predict(padded_sequences)
    pred_class = label_encoder.inverse_transform([pred.argmax(axis=-1)[0]])
    return pred_class[0]

def get_prediction():
    input_text = text_input.get("1.0", "end-1c")
    word_count = len(input_text.split())

    if word_count < 5:
        prediction_label.config(text="Please enter at least 5 words !!!", foreground="orange")
        sentiment_image.config(image=img_blue)  # Show the blue image
    else:
        predict_button.config(text="Analyzing...")
        root.update_idletasks()
        root.after(800, perform_analysis, input_text)  # Delay for analysis

def perform_analysis(input_text):
    prediction = predict(input_text).upper()

    # Update button and prediction label
    predict_button.config(text="Predict Sentiment")
    prediction_label.config(text=f"Predicted Sentiment: {prediction}")

    # Change label color and image based on sentiment
    if prediction == 'POSITIVE':
        prediction_label.config(foreground="green")
        sentiment_image.config(image=img_positive)  # Update image
    elif prediction == 'NEGATIVE':
        prediction_label.config(foreground="red")
        sentiment_image.config(image=img_negative)  # Update image
    else:
        prediction_label.config(foreground="gray")
        sentiment_image.config(image=img_neutral)  # Update image

def resize_image(image_path, max_size):
    """Resize an image while maintaining its aspect ratio."""
    image = Image.open(image_path)
    image.thumbnail(max_size)  # ANTIALIAS filter is no longer needed
    return ImageTk.PhotoImage(image)

# Initialize the main window
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("500x600")
root.resizable(False, False)

# Font customization
custom_font = Font(family="Helvetica", size=12, weight="bold")

# Resize and load images for different sentiments
max_image_size = (100, 100)  # Maximum width and height of the images
img_positive = resize_image("emoji_face\\positive.png", max_image_size)
img_negative = resize_image("emoji_face\\negative.png", max_image_size)
img_neutral = resize_image("emoji_face\\neutral.png", max_image_size)
img_blue = resize_image("emoji_face/BadSmiley.png", max_image_size)
img_blueblue = resize_image("emoji_face/blue.png", max_image_size)

# Label to display sentiment image
sentiment_image = ttk.Label(root, image=img_blueblue)
sentiment_image.pack(pady=20)

# Text input area
text_input = tk.Text(root, height=10, width=50, font=custom_font)
text_input.pack(pady=20)

# Predict button
style = ttk.Style()
style.configure("TButton", font=custom_font)
predict_button = ttk.Button(root, text="Predict Sentiment", command=get_prediction, style="TButton", width=20)
predict_button.pack(pady=10)

# Label to display prediction with custom font
prediction_label = ttk.Label(root, text="Predicted Sentiment: None", font=custom_font)
prediction_label.pack(pady=10)

root.mainloop()
