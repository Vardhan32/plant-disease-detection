# Plant Leaf Disease Detection

A React + Flask web application that classifies an uploaded plant-leaf image using a TensorFlow/Keras image-classification model. The backend returns a predicted class and confidence score; the frontend can read the prediction aloud using the browser's built-in speech synthesis.

> **Project attribution:** This repository is adapted from [Chandini7203/plant-disease-app](https://github.com/Chandini7203/plant-disease-app). The training script structure, Flask inference code, and React component are derived from that upstream project, with local modifications (15-class retraining, `/health` endpoint, hardened training/inference code, improved UI). It is a fork/adaptation, not an entirely original implementation.

> **License note:** The upstream repository contains no license file (checked October 2026), so no redistribution rights are granted there by default. Do not assume attribution alone permits reuse — confirm licensing with the repository owner before redistributing this code or the trained model. No license is added here; that decision is left to the owner.

## Features

- Upload a plant-leaf image in the React interface (JPEG, PNG, WebP, BMP; up to 8 MB).
- Send the image to a Flask `/predict` endpoint.
- Resize the image to 128 × 128 and run model inference.
- Display the predicted class and confidence score, with a note that confidence is not a guarantee of correctness.
- Optionally speak the prediction using the browser's built-in `SpeechSynthesis` API.
- Display general plant-care precautions.
- `GET /health` endpoint reporting backend status and class count.

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
  test_app.py
  saved_model/
    plant_disease_model.h5  # Tracked 15-class model; runs without retraining
    class_names.json        # Class labels in model output-index order
    training_metrics.json   # Training/validation metrics from the last run
frontend/
  src/
  index.html
  .env.example
requirements.txt
README.md
```

## Run locally

### 1. Clone this repository

```bash
git clone https://github.com/Vardhan32/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Dataset

The `dataset/PlantVillage/` class folders (15 classes, ~20,600 images) are already committed in this repository's history, so a fresh clone includes the training data and no download step is needed. The layout is:

```
dataset/PlantVillage/
  Pepper__bell___Bacterial_spot/
  Pepper__bell___healthy/
  Potato___Early_blight/
  Potato___healthy/
  Potato___Late_blight/
  Tomato_Bacterial_spot/
  Tomato_Early_blight/
  Tomato_Late_blight/
  Tomato_Leaf_Mold/
  Tomato_Septoria_leaf_spot/
  Tomato_Spider_mites_Two_spotted_spider_mite/
  Tomato__Target_Spot/
  Tomato__Tomato_YellowLeaf__Curl_Virus/
  Tomato__Tomato_mosaic_virus/
  Tomato_healthy/
```

### 3. Install Python dependencies and run the backend tests

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s backend -p "test_*.py" -t .
```

The tests load the tracked model (no retraining) and cover `/health`, valid-image prediction, missing/empty/invalid uploads, and class-mapping consistency.

### 4. Start the backend

From the repository root:

```bash
python backend/app.py
```

The Flask app runs on `http://127.0.0.1:5000` by default. Debug mode is off unless `FLASK_DEBUG=1` is set. Optional environment variables:

- `PORT` — backend port (default `5000`)
- `MAX_UPLOAD_MB` — maximum request size (default `8`)
- `CORS_ORIGINS` — comma-separated allowed origins (default `*`, suitable for local development)
- `FLASK_DEBUG=1` — enable the Flask debugger (local use only)

### 5. Start the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`). To point the UI at a non-default backend, copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_URL`. To verify the production bundle builds, run `npm run build` from `frontend/`.

## Retraining (local, manual)

Retraining is a deliberate local step — it is not run automatically on push. The CI workflow (`.github/workflows/ci.yml`) only runs backend tests and the frontend production build. A separate manual workflow (`.github/workflows/retrain-model.yml`, launched from the repository's **Actions** tab) can retrain on a GitHub-hosted runner using the committed dataset, with no downloads or credentials required.

To retrain locally, from the repository root:

```bash
pip install -r requirements.txt
python backend/train.py
```

Training uses an 80/20 training/validation split and runs for 5 epochs (~20 minutes on CPU; downloads ~16 MB EfficientNetB0 ImageNet weights on first run). On completion it validates the new artifacts (model output count, class mapping, and metrics must agree) and only then replaces:

- `backend/saved_model/plant_disease_model.h5` — trained model
- `backend/saved_model/class_names.json` — class names in the exact output-index order used during training
- `backend/saved_model/training_metrics.json` — class/sample counts and final-epoch training/validation metrics

Keep these generated artifacts together; the inference app checks that the model output count matches the saved class-name mapping. The pre-retraining backup (`plant_disease_model_backup.h5`) is a local file and is never committed.

## Model and evaluation details

- **Model:** EfficientNetB0 pretrained on ImageNet, frozen as a feature extractor, followed by global average pooling, a 128-unit ReLU dense layer, dropout (0.3), and a softmax output layer.
- **Input size:** 128 × 128 RGB.
- **Input preprocessing:** Training (`train.py`) and inference (`app.py`) both pass RGB pixel values in the `[0, 255]` range. EfficientNetB0 includes its own rescaling layer, so neither divides pixels by 255.
- **Optimizer:** Adam with learning rate 0.0001.
- **Loss:** Categorical cross-entropy.
- **Epochs:** 5.
- **Classes (15):** Pepper bell bacterial spot and healthy; Potato early blight, late blight, and healthy; Tomato bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot, spider mites, target spot, mosaic virus, yellow leaf curl virus, and healthy. See `backend/saved_model/class_names.json` for the exact output-index order.
- **Image augmentation:** No random flips, rotations, zoom, or other augmentation is currently applied.
- **Early stopping:** Not implemented.
- **Accuracy:** The last training run recorded final training accuracy 0.862 and **validation accuracy 0.896** (see `training_metrics.json`). Validation accuracy is not independent test accuracy; evaluate on a separate held-out test set before presenting a final performance claim. Do not infer accuracy from a handful of sample predictions.

### Count the images in your local dataset

Run this from the repository root:

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

- The model covers only the 15 listed classes (pepper, potato, tomato). Images of other crops or non-leaf content will still receive one of these labels.
- Misclassification happens: during local testing, a Potato late-blight leaf was predicted as Tomato late blight at 0.52 confidence. Treat low-confidence predictions with skepticism and confirm important diagnoses independently.
- The frontend default backend URL (`http://127.0.0.1:5000`) suits local development; configure `VITE_API_URL` for any other setup.
- The displayed precautions are general plant-care advice, not disease-specific expert guidance.
- A validation split is used during training, but this repository does not provide an independent test-set evaluation.
- This is a fork/adaptation of an existing project; do not claim the upstream implementation as entirely your own.
- No license file is present; redistribution rights are unresolved (see the license note above).
