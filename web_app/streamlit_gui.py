import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Cache the model, tokenizer, and label encoder to improve performance
@st.cache_resource
def load_model_and_resources():
    model = tf.keras.models.load_model('web_app/final_model_load/final_sentiment_model.h5')
    with open('web_app/final_model_load/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    with open('web_app/final_model_load/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    return model, tokenizer, label_encoder

# Load model, tokenizer, and label encoder
model, tokenizer, label_encoder = load_model_and_resources()

# Function to predict sentiment
def predict(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=256)
    pred = model.predict(padded_sequences)
    pred_class = label_encoder.inverse_transform([pred.argmax(axis=-1)[0]])
    return pred_class[0]

# Cache and resize images to improve performance
@st.cache_data
def load_and_resize_image(image_path, size=(100, 100)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image

# Load images
img_positive = load_and_resize_image("web_app/emoji_face/positive.png")
img_negative = load_and_resize_image("web_app/emoji_face/negative.png")
img_neutral = load_and_resize_image("web_app/emoji_face/neutral.png")
img_blueblue = load_and_resize_image("web_app/emoji_face/blue.png")

# Streamlit interface
st.title("Rojak Language Sentiment Analysis")
st.markdown("### Enter your text below (at least 5 words):")

input_text = st.text_area("", height=150).strip()

st.markdown("Click the button below to predict the sentiment of the entered text:")

if st.button("Predict Sentiment"):
    cleaned_text = ' '.join(input_text.split())  # Remove extra spaces
    if len(cleaned_text.split()) < 5:
        st.warning("Please enter at least 5 words!")
        st.image(img_blueblue, width=100)
    else:
        with st.spinner('Analyzing...'):
            prediction = predict(cleaned_text).upper()
        st.success(f"Predicted Sentiment: **{prediction}**")

        if prediction == 'POSITIVE':
            st.image(img_positive, width=100)
        elif prediction == 'NEGATIVE':
            st.image(img_negative, width=100)
        else:
            st.image(img_neutral, width=100)
