from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\Admin\OneDrive\Desktop\Gen_AI_Work\Project Work\Projects AI and GenAI\Solar Panel Dust Detection\Solar_Model1.0\uploads'
img_height = 128
img_width = 128

# Load labels
def load_label_map(labels_path):
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()
    return labels

# Preprocess image
def preprocess_image(img):
    img_array = cv2.resize(img, (img_height, img_width))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the CNN model
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], request.files['model'].filename)
    model = load_model(model_path)

    # Load labels
    labels_path = os.path.join(app.config['UPLOAD_FOLDER'], request.files['labels'].filename)
    labels = load_label_map(labels_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        img = preprocess_image(frame)

        # Perform prediction
        predictions = model.predict(img)
        predicted_label = labels[np.argmax(predictions)]

        # Display prediction
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
