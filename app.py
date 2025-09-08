import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

# --- Configuration ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_tree_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.pkl")

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = 128  # Match your training input size

# --- Load model ---
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Load class names ---
try:
    with open(CLASS_NAMES_PATH, "rb") as f:
        CLASS_NAMES = pickle.load(f)
    print("Class names loaded successfully!")
except Exception as e:
    print(f"Error loading class names: {e}")
    CLASS_NAMES = []

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not CLASS_NAMES:
        return jsonify({"error": "Model or class names not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[class_index]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence*100, 2)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
