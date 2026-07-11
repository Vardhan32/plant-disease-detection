import tensorflow as tf

model = tf.keras.models.load_model("saved_model/plant_disease_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("saved_model/plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite format.")
