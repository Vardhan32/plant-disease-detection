# Plant Leaf Disease Detection

A React + Flask web application that classifies an uploaded plant-leaf image using a TensorFlow/Keras image-classification model. The backend returns a predicted class and confidence score; the frontend can read the prediction aloud using the browser's built-in speech synthesis.

> **Project attribution:** This repository is based on [Chandini7203/plant-disease-app](https://github.com/Chandini7203/plant-disease-app). The README, training script, Flask inference code, and React component retain corresponding upstream implementations. I have not verified substantial original changes to those core files, so describe this honestly as a fork/adaptation rather than entirely original work. Document future changes clearly.

## Features

- Upload a plant-leaf image in the React interface.
- Send the image to a Flask `/predict` endpoint.
- Resize the image to 128 × 128 and run model inference.
- Display the predicted class and confidence score.
- Optionally speak the prediction using the browser's built-in `SpeechSynthesis` API.
- Display general plant-care precautions.

## Tech stack

- **Frontend:** React, Vite, JavaScript, Axios
- **Backend:** Python, Flask, Flask-CORS
- **ML:** TensorFlow/Keras, EfficientNetB0 pretrained on ImageNet, NumPy
- **Image handling:** Pillow (PIL)
- **Voice output:** Browser `SpeechSynthesis` API — the current frontend does not use gTTS
- **Dataset source:** [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

## Repository layout

```
backend/
  app.py
  train.py
  convert_model.py
  saved_model/          # Generated locally by train.py
frontend/
  src/
  index.html
requirements.txt
README.md
```

The training script expects the dataset at `dataset/PlantVillage/`, with one subdirectory per class. The dataset and trained model are not bundled in this repository.

## Run locally

### 1. Clone this repository

```bash
git clone https://github.com/Vardhan32/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Prepare the dataset

Place the PlantVillage class folders in `dataset/PlantVillage/`. The directory should look like this:

```
dataset/PlantVillage/
  Apple___Black_rot/
  Apple___healthy/
  ...
```

Each class folder should contain its corresponding images.

### 3. Install Python dependencies and train the model

From the repository root:

```bash
pip install -r requirements.txt
python backend/train.py
```

Training uses an 80/20 training/validation split and runs for up to 5 epochs. On completion, it generates these local files:

- `backend/saved_model/plant_disease_model.h5` — trained model
- `backend/saved_model/class_names.json` — class names in the exact output-index order used during training
- `backend/saved_model/training_metrics.json` — class/sample counts and final-epoch training/validation metrics

The model is not automatically uploaded to GitHub. Keep these generated artifacts together; the inference app checks that the model output count matches the saved class-name mapping.

### 4. Start the backend

From the repository root:

```bash
python backend/app.py
```

The Flask app runs on `http://127.0.0.1:5000` by default. You can also run it from the backend directory with `python app.py`.

### 5. Start the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`).

## Model and evaluation details

- **Model:** EfficientNetB0 pretrained on ImageNet, frozen as a feature extractor, followed by global average pooling, a 128-unit ReLU dense layer, dropout (0.3), and a softmax output layer.
- **Input size:** 128 × 128 RGB.
- **Input preprocessing:** The current training and inference code passes RGB pixel values in the `[0, 255]` range. EfficientNetB0 includes its own rescaling layer, so the code does not divide the pixels by 255 before passing them to the model.
- **Optimizer:** Adam with learning rate 0.0001.
- **Loss:** Categorical cross-entropy.
- **Epochs:** Up to 5.
- **Class labels:** Generated from the dataset directory names during training and saved to `class_names.json`; the backend loads that same mapping instead of relying on a separate hard-coded list.
- **Image augmentation:** No random flips, rotations, zoom, or other augmentation is currently applied.
- **Early stopping:** Not implemented.
- **Accuracy:** No accuracy figure is claimed here. After training, inspect `backend/saved_model/training_metrics.json` for the final training and validation metrics. Validation accuracy is not the same as independent test accuracy; evaluate on a separate held-out test set before presenting a final performance claim.

### Count the images in your local dataset

The exact count depends on the dataset copy you use. Run this from the repository root:

```python
from pathlib import Path

root = Path("dataset/PlantVillage")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not root.is_dir():
    raise SystemExit(f"Dataset directory not found: {root.resolve()}")

counts = {
    folder.name: sum(
        1 for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )
    for folder in root.iterdir()
    if folder.is_dir()
}

for class_name, count in sorted(counts.items()):
    print(f"{class_name}: {count}")

print(f"Classes: {len(counts)}")
print(f"Total images: {sum(counts.values())}")
```

Record the dataset version, class count, and total image count when documenting model results.

## Current limitations

- The model is trained on the dataset you provide locally; results depend on dataset quality and class coverage.
- The frontend currently points to `http://127.0.0.1:5000/predict`, suitable for local development rather than a deployed service.
- The displayed precautions are general plant-care advice, not disease-specific expert guidance.
- A validation split is used during training, but this repository does not provide an independent test-set evaluation.
- This is a fork/adaptation of an existing project; do not claim the upstream implementation as entirely your own.

## Screenshots

Add screenshots using their actual repository paths when available. Avoid broken image links.
