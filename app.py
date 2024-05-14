from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\Admin\OneDrive\Desktop\Gen_AI_Work\Project Work\Projects AI and GenAI\Solar Panel Dust Detection\Solar_Model1.0\uploads'

def load_label_map(labels_path):
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()
    return labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the CNN model
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], request.files['model'].filename)
    model = load_model(model_path)

    # Load labels
    labels_path = os.path.join(app.config['UPLOAD_FOLDER'], request.files['labels'].filename)
    labels = load_label_map(labels_path)

    # Load and preprocess image
    img = Image.open(request.files['image'])
    img = img.resize((128, 128))  # Assuming input shape required by your CNN model
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize and add batch dimension

    # Perform prediction
    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions)]

    return render_template('index.html', predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
