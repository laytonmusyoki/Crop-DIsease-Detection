import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model("models/crop_disease_model.h5")

# Load class names (update if hardcoded or stored elsewhere)
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Function to load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the image for prediction
image_path = "test.jpeg"  

# Predict
try:
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    print("Predicted index:", np.argmax(prediction))


    print(f"Predicted disease class: {predicted_class}")
except Exception as e:
    print(f"Error: {e}")
