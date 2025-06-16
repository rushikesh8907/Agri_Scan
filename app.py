from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = load_model('model/plant_disease_model.h5')
CLASS_NAMES = ['Apple___Black_rot', 'Apple___healthy', 'Corn___Common_rust', 'Corn___healthy']

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/')
def index():
    return "AI-Based Plant Disease Scanner is running."

@app.route('/scan')
def scan():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
