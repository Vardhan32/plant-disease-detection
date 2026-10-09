import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Resolve paths from this file so the script works from the repo root or backend/.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
DATA_DIR = REPO_DIR / "dataset" / "PlantVillage"
MODEL_DIR = BACKEND_DIR / "saved_model"
MODEL_PATH = MODEL_DIR / "plant_disease_model.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}. "
            "Place the PlantVillage class folders inside dataset/PlantVillage/."
        )

    class_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class subdirectories in {DATA_DIR}, "
            f"but found {len(class_dirs)}."
        )

    # EfficientNetB0 includes its own rescaling layer and expects RGB pixel
    # values in the [0, 255] range. Do not divide by 255 here.
    datagen = ImageDataGenerator(validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=42,
    )
    val_gen = datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    if train_gen.class_indices != val_gen.class_indices:
        raise RuntimeError("Training and validation class mappings do not match.")

    num_classes = train_gen.num_classes
    class_names = [
        name for name, index in sorted(
            train_gen.class_indices.items(), key=lambda item: item[1]
        )
    ]

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save to temporary paths first so an interrupted run never leaves a
    # half-written model. Only replace the active files after validation.
    tmp_model_path = MODEL_DIR / "plant_disease_model_new.h5"
    tmp_classes_path = MODEL_DIR / "class_names_new.json"
    tmp_metrics_path = MODEL_DIR / "training_metrics_new.json"

    model.save(str(tmp_model_path))

    with tmp_classes_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2, ensure_ascii=False)

    final_metrics = {
        metric: float(values[-1])
        for metric, values in history.history.items()
        if values
    }
    metrics_record = {
        "epochs_completed": len(history.epoch),
        "input_size": list(IMG_SIZE),
        "class_count": num_classes,
        "class_names": class_names,
        "train_samples": int(train_gen.samples),
        "validation_samples": int(val_gen.samples),
        "final_epoch_metrics": final_metrics,
        "note": (
            "Validation metrics are from the 20% validation split, not an "
            "independent test set. No image augmentation or early stopping "
            "is used by this script."
        ),
    }
    with tmp_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_record, file, indent=2, ensure_ascii=False)

    # Validate the new artifacts match each other before replacing the
    # active model. Both pre-existing .h5 files are left untouched until here.
    reloaded = tf.keras.models.load_model(str(tmp_model_path))
    if int(reloaded.output_shape[-1]) != len(class_names):
        raise ValueError(
            f"New model outputs {reloaded.output_shape[-1]} classes but "
            f"mapping has {len(class_names)} labels. Active model not replaced."
        )
    if metrics_record["class_count"] != len(class_names):
        raise ValueError("Training metrics class count does not match class mapping.")
    if metrics_record["class_names"] != class_names:
        raise ValueError("Training metrics class names do not match class mapping.")
    del reloaded
    tmp_model_path.replace(MODEL_PATH)
    tmp_classes_path.replace(CLASS_NAMES_PATH)
    tmp_metrics_path.replace(METRICS_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class mapping saved to: {CLASS_NAMES_PATH}")
    print(f"Training metrics saved to: {METRICS_PATH}")
    print(f"Classes: {num_classes}; class names: {class_names}")
    print(f"Final epoch metrics: {final_metrics}")


if __name__ == "__main__":
    main()
