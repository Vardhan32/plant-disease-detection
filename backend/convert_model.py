"""Optional: convert the active Keras model to TensorFlow Lite.

Run from the repository root:

    python backend/convert_model.py

The Flask application serves the Keras HDF5 model directly; this script is only
for experiments that need a smaller TFLite artifact for mobile or edge use.
"""

from pathlib import Path

import tensorflow as tf

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_DIR / "backend" / "saved_model" / "plant_disease_model.h5"
OUTPUT_PATH = MODEL_PATH.with_suffix(".tflite")

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)
print(f"Wrote TFLite model to {OUTPUT_PATH}")
