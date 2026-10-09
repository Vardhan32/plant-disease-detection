# Plant Leaf Disease Detection

A React + Flask web application that classifies an uploaded plant-leaf image using a TensorFlow/Keras image-classification model. The backend returns a predicted class and confidence score; the frontend can read the prediction aloud using the browser's built-in speech synthesis.

> **Project attribution:** This repository is based on [Chandini7203/plant-disease-app](https://github.com/Chandini7203/plant-disease-app). The current README, training script, Flask inference code, and React component retain the corresponding upstream implementations. I have not verified any substantial original changes to those core files, so this project should be described as a fork/adaptation rather than entirely original work. Any future changes should be documented here clearly.

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
- **Voice output:** Browser `SpeechSynthesis` API — no gTTS dependency is used by the current frontend
- **Dataset source:** [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

## Repository layout

```
backend/
  app.py
  train.py
  convert_model.py
  saved_model/
frontend/
  src/
  index.html
requirements.txt
README.md
```

The training script expects the dataset at `dataset/PlantVillage/`, with one subdirectory per class. The dataset itself is not described by a reproducible image-count manifest in this repository.

## Run locally

### 1. Clone this repository

```bash
git clone https://github.com/Vardhan32/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Start the backend

From the repository root:

```bash
pip install -r requirements.txt
cd backend
python app.py
```

The Flask app runs on `http://127.0.0.1:5000` by default. The saved model must exist at `backend/saved_model/plant_disease_model.h5` before starting the app.

### 3. Start the frontend

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
- **Optimizer:** Adam with learning rate 0.0001.
- **Loss:** Categorical cross-entropy.
- **Epochs in the current training script:** 5.
- **Class labels in inference code:** 10 hard-coded labels in `backend/app.py`. The training script derives the output size from the number of subdirectories in the dataset, so verify that the dataset contains exactly these 10 classes and that the class order matches before using the saved model.
- **Image augmentation:** The current training script rescales pixel values and creates an 80/20 training/validation split. It does **not** currently apply random flips, rotations, zoom, or other image augmentation.
- **Early stopping:** Not implemented in the current training script.
- **Accuracy:** Not reported here. An earlier README claimed approximately 94%, but no training log or evaluation artifact is committed to show whether that was training or validation accuracy. Do not quote 94% as a verified result unless you can reproduce it and record the relevant metric.

### Count the images in your local dataset

The image count depends on the exact dataset copy placed in `dataset/PlantVillage`. Run this from the repository root to count common image formats by class and in total:

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

Record the resulting class count and image total alongside the dataset version used for training. For evaluation, keep the validation accuracy from the final training epoch separate from training accuracy, and preferably evaluate on a held-out test set that was not used for training or model selection.

## Current limitations to address

- The backend's class labels are hard-coded. Keep them synchronized with the training generator's class-to-index mapping.
- The frontend currently points to `http://127.0.0.1:5000/predict`; this is suitable for local development, not a deployed service.
- The displayed precautions are general plant-care advice, not disease-specific expert guidance.
- The repository does not currently provide a committed training log, dataset manifest, or independent test evaluation.

## Screenshots

If the screenshots are present in the repository, add them here using their actual paths. Avoid broken image links.
