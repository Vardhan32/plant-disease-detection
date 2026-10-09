import io
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image


app = Flask(__name__)

# Upload guard: reject overly large request bodies before image parsing.
# Configurable via MAX_UPLOAD_MB; defaults to 8 MB for local development.
try:
    _max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "8"))
except ValueError:
    _max_upload_mb = 8
app.config["MAX_CONTENT_LENGTH"] = _max_upload_mb * 1024 * 1024

# CORS: permissive by default for local development (Vite on localhost:5173
# calling 127.0.0.1:5000). Restrict in other environments with a
# comma-separated allow-list, e.g. CORS_ORIGINS="https://my-frontend.example".
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
CORS(app, resources={r"/*": {"origins": _cors_origins or "*"}})

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "saved_model" / "plant_disease_model.h5"
CLASS_NAMES_PATH = BACKEND_DIR / "saved_model" / "class_names.json"

# Input size must match backend/train.py IMG_SIZE. EfficientNetB0 contains its
# own rescaling layer, so both training and inference pass RGB pixels in
# [0, 255] without dividing by 255.
IMG_SIZE = (128, 128)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run train.py from the backend directory first."
    )
if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Class mapping not found at {CLASS_NAMES_PATH}. "
        "Retrain with the current backend/train.py to generate it."
    )

model = tf.keras.models.load_model(str(MODEL_PATH))
with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError(f"Invalid or empty class mapping in {CLASS_NAMES_PATH}")
if any(not isinstance(name, str) or not name for name in class_names):
    raise ValueError(f"Class mapping in {CLASS_NAMES_PATH} must be a list of names")

model_output_classes = int(model.output_shape[-1])
if len(class_names) != model_output_classes:
    raise ValueError(
        f"Class mapping has {len(class_names)} labels but model outputs "
        f"{model_output_classes} classes. Retrain the model and restart the app."
    )


@app.errorhandler(413)
def request_too_large(_exc):
    return (
        jsonify(
            {
                "error": (
                    f"Uploaded file is too large. "
                    f"Maximum request size is {_max_upload_mb} MB."
                )
            }
        ),
        413,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(class_names)})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    try:
        raw = file.read()
        if not raw:
            return jsonify({"error": "Uploaded file is empty"}), 400
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image = image.resize(IMG_SIZE)
        # Match training: EfficientNetB0 contains its own input rescaling.
        # Pass RGB pixel values in [0, 255]; do not divide by 255 here.
        img_array = np.asarray(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        return jsonify(
            {
                "prediction": class_names[predicted_index],
                "confidence": float(predictions[predicted_index]),
            }
        )
    except (OSError, ValueError):
        # Includes unidentifiable/corrupt image data. Keep the message
        # generic so internal details are not leaked to clients.
        return jsonify({"error": "Could not process the uploaded image"}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the backend logs."}), 500


if __name__ == "__main__":
    # Debug mode is opt-in only: set FLASK_DEBUG=1 for local debugging.
    # Never enable the debugger when exposed beyond localhost.
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
