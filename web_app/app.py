import os
from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model and tokenizer/label_encoder only once and reuse them
model = tf.keras.models.load_model('final_model_load/final_sentiment_model.h5', compile=False)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    
    if request.method == 'POST':
        input_text = request.form['text_input']
        if len(input_text.split()) < 5:
            prediction = "Please enter at least 5 words!"
            image_url = "/static/img/blue.png"
        else:
            prediction = predict(input_text).upper()
            if prediction == 'POSITIVE':
                image_url = "/static/img/positive.png"
            elif prediction == 'NEGATIVE':
                image_url = "/static/img/negative.png"
            else:
                image_url = "/static/img/neutral.png"
    
    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT if available, otherwise default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)
