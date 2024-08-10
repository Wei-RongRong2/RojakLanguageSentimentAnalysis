import tensorflow as tf
from tensorflow.keras.layers import LSTM

# Define custom objects if any (this is just an example)
custom_objects = {
    "LSTM": LSTM
}

# Load the model with custom objects
model = tf.keras.models.load_model('web_app/final_model_load/final_sentiment_model.h5', custom_objects=custom_objects, compile=False)
