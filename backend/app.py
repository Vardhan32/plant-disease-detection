import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image


app = Flask(__name__)
CORS(app)

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "saved_model" / "plant_disease_model.h5"
CLASS_NAMES_PATH = BACKEND_DIR / "saved_model" / "class_names.json"

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

model_output_classes = int(model.output_shape[-1])
if len(class_names) != model_output_classes:
    raise ValueError(
        f"Class mapping has {len(class_names)} labels but model outputs "
        f"{model_output_classes} classes. Retrain the model and restart the app."
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
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((128, 128))
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
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"Could not process the uploaded image: {exc}"}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the backend logs."}), 500


if __name__ == "__main__":
    app.run(debug=True)
