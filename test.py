import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# Load model architecture
with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)

# Load saved weights
model.load_weights("os/model_weights.h5")

# Define class labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load and preprocess a test image
img_path = "test_img.jpg"  
img_size = 48

img = load_img(img_path, color_mode="grayscale", target_size=(img_size, img_size))
img_array = img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make a prediction
predictions = model.predict(img_array)
predicted_class = class_labels[np.argmax(predictions)]

print(f"Predicted Emotion: {predicted_class}")