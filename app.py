import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
from skimage.feature import hog
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained models
cnn_model = load_model("model/cnn_model.h5")
svm_model = joblib.load("model/svm_model.joblib")

# HOG feature extractor
def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return render_template("index.html", label=None, confidence=None, error="Please upload an image!")

        # Save image
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # CNN Prediction
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0) / 255.0
        cnn_pred = cnn_model.predict(img)[0][0]

        # SVM Prediction
        hog_features = extract_hog_features(img_path).reshape(1, -1)
        svm_pred = svm_model.predict_proba(hog_features)[0][1]

        # Combine predictions
        avg_confidence = float((cnn_pred + svm_pred) / 2)
        label = "Fake" if avg_confidence > 0.5 else "Real"

        return render_template(
            "index.html",
            label=label,
            confidence=round(avg_confidence * 100, 2),
            image_path=img_path.replace("\\", "/")
        )

    return render_template("index.html", label=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
