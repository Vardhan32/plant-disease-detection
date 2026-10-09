# Plant Leaf Disease Detection

A web application that predicts plant-leaf disease classes from an uploaded image. It uses a React frontend, a Flask API, and a TensorFlow/Keras image-classification model. The interface displays the predicted class and confidence score and can read the result aloud using the browser's built-in speech synthesis.

> **Project note:** This repository adapts an existing plant-disease detection implementation and includes local changes to the model classes, inference API, validation, and user interface. It should be described as an adapted project, not as an entirely original implementation.
>
> **Licensing:** The original source repository did not provide a license file when checked in October 2026. Permission to redistribute code or model artifacts may therefore be unresolved. Attribution alone does not grant reuse rights. Confirm the applicable permissions before redistributing this project.

## Features

- Upload a leaf image in JPEG, PNG, WebP, or BMP format (maximum 8 MB).
- Predict a class through the Flask `/predict` API.
- Display the predicted class and model confidence.
- Read the prediction aloud with the browser's `SpeechSynthesis` API.
- Show general plant-care precautions.
- Check backend status and supported class count through `GET /health`.
- Run backend tests and build the frontend in CI.

## Technology stack

| Area | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, Axios |
| Backend | Python, Flask, Flask-CORS |
| Machine learning | TensorFlow/Keras, EfficientNetB0, NumPy |
| Image processing | Pillow |
| Dataset | [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |

## How it works

1. A user uploads a plant-leaf image in the web interface.
2. The frontend sends the image to the Flask API.
3. The backend converts the image to RGB and resizes it to 128 × 128 pixels.
4. The trained model predicts one of the 15 supported classes.
5. The interface displays the class and confidence score, with an optional spoken result.

**Important:** The confidence score is not a guarantee that the prediction is correct. Use the result as an initial indication, not as a definitive diagnosis.

## Supported classes

The model supports 15 classes across pepper, potato, and tomato:

- **Pepper:** bacterial spot, healthy
- **Potato:** early blight, late blight, healthy
- **Tomato:** bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot, spider mites, target spot, yellow leaf curl virus, mosaic virus, healthy

The exact class-to-output-index mapping is stored in `backend/saved_model/class_names.json`.

## Run locally

### Prerequisites

- Python and pip
- Node.js and npm

### 1. Clone the repository

```bash
git clone https://github.com/Vardhan32/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install Python dependencies

Run these commands from the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s backend -p "test_*.py" -t .
```

The backend tests cover the health endpoint, image prediction, invalid or missing uploads, and consistency of the class mapping. They use the tracked model, so retraining is not required to run them.

### 3. Start the backend

From the repository root:

```bash
python backend/app.py
```

The API runs at `http://127.0.0.1:5000` by default.

Optional environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `PORT` | Backend port | `5000` |
| `MAX_UPLOAD_MB` | Maximum upload size in MB | `8` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |
| `FLASK_DEBUG` | Enable Flask debug mode for local development | Disabled |

For deployment, configure CORS to allow only the origins you trust.

### 4. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`).

To use a backend at a different URL, copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_URL`. To verify the production frontend build, run:

```bash
npm run build
```

## Model details and evaluation

- **Architecture:** EfficientNetB0 pretrained on ImageNet, used as a frozen feature extractor, followed by global average pooling, a 128-unit ReLU dense layer, dropout (0.3), and a 15-class softmax output.
- **Input:** RGB image resized to 128 × 128.
- **Optimizer:** Adam, learning rate 0.0001.
- **Loss:** Categorical cross-entropy.
- **Training:** 5 epochs with an 80/20 training-validation split.
- **Augmentation:** Random flips, rotations, and zoom are not currently applied.
- **Evaluation caveat:** The last recorded run reported training accuracy of 0.862 and validation accuracy of 0.896. Validation accuracy is not independent test accuracy; evaluate on a separate held-out test set before making stronger performance claims. Results can vary between runs.

EfficientNetB0 includes its own input rescaling layer. The training and inference code pass RGB pixel values in the 0–255 range without dividing them by 255.

### Retrain the model (optional)

The repository includes a saved model, so retraining is not needed for normal local use. To train again, run from the repository root:

```bash
pip install -r requirements.txt
python backend/train.py
```

Training may download the pretrained EfficientNetB0 weights on the first run and can take time on a CPU. The training script validates the generated model artifacts before replacing the saved model, class mapping, and metrics. Keep these files together:

- `backend/saved_model/plant_disease_model.h5`
- `backend/saved_model/class_names.json`
- `backend/saved_model/training_metrics.json`

Retraining is separate from the normal CI checks. The CI workflow tests the backend and builds the frontend; a manual retraining workflow may also be available in the GitHub Actions tab.

## Repository structure

```text
backend/
  app.py
  train.py
  convert_model.py
  test_app.py
  saved_model/
    plant_disease_model.h5
    class_names.json
    training_metrics.json
frontend/
  src/
  index.html
  .env.example
requirements.txt
README.md
```

## Limitations

- Only the 15 listed pepper, potato, and tomato classes are supported. Other crops, unrelated images, or poor-quality photos may still be assigned an available class.
- Predictions can be incorrect, including confusion between visually similar diseases. Confirm important decisions with a qualified agricultural expert.
- The repository does not currently provide an independent held-out test-set evaluation.
- Plant-care precautions are general information and are not a substitute for disease-specific expert advice.
- The application is a learning project and should not be treated as a production-grade diagnostic system.

## Future improvements

- Evaluate on an independent test set and report per-class precision, recall, and a confusion matrix.
- Add confidence thresholds and a clear fallback for unsupported or low-quality images.
- Apply and compare appropriate image augmentation strategies.
- Improve accessibility, mobile responsiveness, and deployment configuration.
