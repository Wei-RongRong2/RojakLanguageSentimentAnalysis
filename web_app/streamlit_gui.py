import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_models():
    # Load models without compiling them
    model_eval = tf.keras.models.load_model('web_app/final_model_load/final_sentiment_model.h5', compile=False)
    return model_eval

# Load the model and tokenizer/label_encoder only once and reuse them
model = load_models()

# Load the tokenizer
with open('web_app/final_model_load/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the label encoder
with open('web_app/final_model_load/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def predict(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=256)
    pred = model.predict(padded_sequences)
    pred_class = label_encoder.inverse_transform([pred.argmax(axis=-1)[0]])
    return pred_class[0]

# Streamlit interface
st.title("Rojak Language Sentiment Analysis")

# Load images
img_positive = Image.open("web_app/emoji_face/positive.png")
img_negative = Image.open("web_app/emoji_face/negative.png")
img_neutral = Image.open("web_app/emoji_face/neutral.png")
img_blueblue = Image.open("web_app/emoji_face/blue.png")

input_text = st.text_area("Enter text (at least 5 words):", height=200)

if st.button("Predict Sentiment"):
    if len(input_text.split()) < 5:
        st.warning("Please enter at least 5 words!")
        st.image(img_blueblue, width=100)
    else:
        prediction = predict(input_text).upper()
        st.write(f"Predicted Sentiment: **{prediction}**")

        if prediction == 'POSITIVE':
            st.image(img_positive, width=100)
        elif prediction == 'NEGATIVE':
            st.image(img_negative, width=100)
        else:
            st.image(img_neutral, width=100)
