from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    model = load_model("models/crop_disease_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_array):
    """Preprocess image for model prediction"""
    try:
        # Resize image to 128x128
        img = cv2.resize(image_array, (128, 128))
        # Normalize pixel values
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def parse_disease_class(class_name):
    """Parse disease class name into readable format"""
    parts = class_name.split('_')
    crop = parts[0]
    is_healthy = 'healthy' in class_name.lower()
    
    disease = ''
    if not is_healthy:
        # Remove crop name and join the rest
        disease_parts = [part for part in parts[1:] if part and part != '']
        disease = ' '.join(disease_parts).replace('_', ' ').title()
    
    return {
        'crop': crop.title(),
        'disease': disease,
        'is_healthy': is_healthy
    }

def get_disease_info(parsed_disease):
    """Get additional information about the disease"""
    if parsed_disease['is_healthy']:
        return {
            'description': f"Your {parsed_disease['crop'].lower()} looks healthy! Continue with regular care and monitoring.",
            'recommendations': [
                "Maintain proper watering schedule",
                "Ensure adequate sunlight",
                "Monitor for any changes in appearance",
                "Regular fertilization as needed"
            ]
        }
    else:
        # Disease-specific information
        disease_info = {
            'Bacterial Spot': {
                'description': "Bacterial spot is a common disease that affects leaves and fruits, causing dark spots and reduced yield.",
                'recommendations': [
                    "Apply copper-based fungicides",
                    "Improve air circulation around plants",
                    "Avoid overhead watering",
                    "Remove affected plant parts"
                ]
            },
            'Early Blight': {
                'description': "Early blight causes dark spots on leaves that may have concentric rings, leading to leaf yellowing and drop.",
                'recommendations': [
                    "Apply fungicide treatments",
                    "Ensure proper plant spacing",
                    "Water at soil level, not on leaves",
                    "Remove infected plant debris"
                ]
            },
            'Late Blight': {
                'description': "Late blight can quickly destroy crops, causing dark patches on leaves and stems.",
                'recommendations': [
                    "Apply preventive fungicides",
                    "Improve air circulation",
                    "Avoid watering in evening",
                    "Remove affected plants immediately"
                ]
            },
            'Leaf Mold': {
                'description': "Leaf mold appears as yellow patches on upper leaf surfaces with fuzzy growth underneath.",
                'recommendations': [
                    "Reduce humidity around plants",
                    "Improve ventilation",
                    "Apply appropriate fungicides",
                    "Remove affected leaves"
                ]
            }
        }
        
        # Get specific info or default
        disease_key = parsed_disease['disease']
        info = disease_info.get(disease_key, {
            'description': f"Disease detected: {disease_key}. Consult with a local agricultural expert for specific treatment.",
            'recommendations': [
                "Consult with a local agricultural expert",
                "Consider appropriate treatment options",
                "Isolate affected plants if necessary",
                "Monitor other plants for similar symptoms"
            ]
        })
        
        return info

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
    
    try:
        # Read and process the image
        image_bytes = file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Preprocess for model
        processed_image = preprocess_image(opencv_image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)
        confidence = float(prediction[0][predicted_index])
        predicted_class = class_names[predicted_index]
        
        # Parse disease information
        parsed_disease = parse_disease_class(predicted_class)
        disease_info = get_disease_info(parsed_disease)
        
        # Prepare response
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'predicted_index': int(predicted_index),
            'confidence': confidence,
            'crop': parsed_disease['crop'],
            'disease': parsed_disease['disease'],
            'is_healthy': parsed_disease['is_healthy'],
            'description': disease_info['description'],
            'recommendations': disease_info['recommendations']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'supported_classes': len(class_names)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)