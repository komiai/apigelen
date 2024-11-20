from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)

# Load class mapping
try:
    with open('class_mapping.json', 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
except Exception as e:
    print("Error loading class mapping:", e)
    class_mapping = {}

# Load model
try:
    model = tf.keras.models.load_model('your_model_0.9.h5')
except Exception as e:
    print("Error loading model:", e)
    model = None

# Directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess image function
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
    # image = ImageOps.grayscale(image)  # Convert image to grayscale
    image = image.resize((150, 150))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict function
def predict_image(image, threshold=0.1):
    if model is None:
        return "Model not loaded"
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        # Get all classes with probabilities above the threshold
        classes_above_threshold = []
        for i, prob in enumerate(prediction[0]):
            if prob > threshold:
                class_label = class_mapping.get(str(i), "Unknown")
                classes_above_threshold.append((class_label, prob))
        
        # Sort classes by probability in descending order
        classes_above_threshold.sort(key=lambda x: x[1], reverse=True)
        
        return classes_above_threshold
    except Exception as e:
        print("Error predicting:", e)
        return "Unknown"
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['imageFile']
    try:
        # Save the uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        image = Image.open(filename)
        predictions = predict_image(image)
        predictions = [[item[0], float(item[1])] for item in predictions]
        return jsonify({'predictions': predictions, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/class_info/<predicted_class_label>')
def class_info(predicted_class_label):
    # Assuming you have a dictionary containing the class information
    # Replace this with your actual data
    class_info = {
        'hiragana_い (i)': {
            'class_name': 'Hiragana い (i)',
            'reading': 'i',
            'meaning': 'none',
            'image_url': '/static/images/hiragana_i.jpeg'  # Replace with the URL of your image
        }
    }
    class_data = class_info.get(predicted_class_label, {})
    
    return render_template('class_info.html', class_name=class_data.get('class_name', 'Unknown'), 
                                              reading=class_data.get('reading', 'Unknown'), 
                                              meaning=class_data.get('meaning', 'Unknown'),
                                              image_url=class_data.get('image_url', ''))


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
