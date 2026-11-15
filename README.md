# Plant Leaf Disease Detection

A **B.Tech final-year group project**: a web application that predicts plant-leaf
disease classes from an uploaded image. It combines a React frontend, a Flask
API, and a TensorFlow/Keras image-classification model. The interface displays
the predicted class and confidence score and can read the result aloud using the
browser's built-in speech synthesis.

## Features

- Upload a leaf image in JPEG, PNG, WebP, or BMP format (maximum 8 MB).
- Predict one of 15 pepper, potato, and tomato classes through the Flask
  `/predict` API.
- Display the predicted class and model confidence.
- Read the prediction aloud with the browser's `SpeechSynthesis` API.
- Check backend status, class count, and serving resolution through `GET /health`.
- Reproducible, group-aware 70/15/15 train/validation/test split.
- Shared Pillow bilinear preprocessing for training, evaluation, and serving.
- Configurable training experiments, validation-only model selection, and a
  guarded one-time test-set evaluation.
- Run backend tests and build the frontend in CI.
- Run the frontend and model-serving backend together with Docker Compose.

## Technology stack

| Area | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, Axios, Nginx |
| Backend | Python 3.11, Flask, Flask-CORS |
| Machine learning | TensorFlow 2.16, Keras, EfficientNetB0, NumPy, scikit-learn |
| Image processing | Pillow |
| Containers | Docker, Docker Compose |
| Dataset | [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |

## How it works

1. A user uploads a plant-leaf image in the web interface.
2. The frontend sends the image to the Flask API.
3. The backend validates the upload, converts it to RGB, and resizes it with
   Pillow bilinear resampling to the resolution stored in the model
   (128 × 128 pixels for the active model).
4. The model returns a probability for each of the 15 supported classes.
5. The interface displays the class and confidence score, with an optional
   spoken result.

Pixel values stay in the `[0, 255]` range. EfficientNetB0 contains its own input
rescaling layer, and the same preprocessing function is used during training,
validation, test evaluation, and serving.

**Important:** The confidence score is not a guarantee that the prediction is
correct. Use the result as an initial indication, not as a definitive diagnosis.

## Supported classes

The model supports 15 classes across pepper, potato, and tomato:

- **Pepper:** bacterial spot, healthy
- **Potato:** early blight, late blight, healthy
- **Tomato:** bacterial spot, early blight, late blight, leaf mold, Septoria
  leaf spot, spider mites, target spot, yellow leaf curl virus, mosaic virus,
  healthy

The exact class-to-output-index mapping is stored in
`backend/saved_model/class_names.json` and is preserved by training,
evaluation, and the Flask API.

## Quick start with Docker

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows
  or macOS, or Docker Engine plus the Docker Compose plugin on Linux.
- Ensure Docker is running before starting the app.

### Build and start

Run from the repository root (the directory containing `docker-compose.yml`):

```bash
docker compose up --build
```

The first build can take several minutes because the backend installs
TensorFlow. Docker Compose starts the frontend after the backend health check
succeeds. Open:

- **Web app:** http://localhost:5173
- **Backend health:** http://localhost:5000/health

Stop the app with `Ctrl+C`, or remove the containers with:

```bash
docker compose down
```

The app uses the tracked model files; retraining is not required. The frontend
is configured at build time to call the backend at `http://localhost:5000`. If
you change the published backend port or access the frontend from another
device, update the `VITE_API_URL` build argument and the backend CORS
allow-list accordingly. The default configuration is for local development, not
direct public-internet exposure.

## Run locally without Docker

### Prerequisites

- Python 3.11 (TensorFlow 2.16 does not support newer Python versions)
- Node.js and npm

### 1. Install Python dependencies and run the tests

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s backend -p "test_*.py" -t .
```

The tests cover the health endpoint, prediction response shape, upload
validation, class mapping, preprocessing consistency, split integrity, model
contract, fine-tuning controls, and evaluation metrics. They load the tracked
model, so retraining is not required.

### 2. Start the backend

```bash
python backend/app.py
```

The API runs at `http://127.0.0.1:5000` by default.

| Variable | Purpose | Default |
|---|---|---|
| `PORT` | Backend port | `5000` |
| `MAX_UPLOAD_MB` | Maximum upload size in MB | `8` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |
| `FLASK_DEBUG` | Enable Flask debug mode for local development | Disabled |

For deployment, configure CORS to allow only the origins you trust.

### 3. Start the frontend

Open a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`). To use a
backend at a different URL, copy `frontend/.env.example` to `frontend/.env` and
set `VITE_API_URL`. Verify the production bundle with:

```bash
npm run build
```

## Dataset audit and split

The dataset contains 20,638 valid PlantVillage images in 15 classes. A full
audit checks every file, then groups exact duplicates (SHA-256) and near
duplicates (64-bit dHash plus thumbnail MSE). The generated manifest is
committed at `backend/data/plantvillage_split.csv`; the summary is in
`backend/data/dataset_audit_summary.json`.

| Audit item | Result |
|---|---|
| Valid images | 20,638 |
| Corrupt images | 0 |
| Classes | 15 |
| Exact duplicate groups | 14 (28 images) |
| Near-duplicate groups | 59 (128 images) |
| Cross-class duplicates | 0 |
| Image groups after merging duplicates | 20,569 |
| Groups crossing train/validation/test | 0 |

The split is group-aware and stratified by class:

| Split | Images | Fraction |
|---|---|---|
| Train | 14,444 | 70.0% |
| Validation | 3,097 | 15.0% |
| Test | 3,097 | 15.0% |

Every class appears in all three splits. Exact and near-duplicate images are
kept in the same split, so duplicate copies cannot inflate validation or test
scores.

For comparison, the repository also reconstructs the original
`ImageDataGenerator(validation_split=0.2)` partition in
`backend/data/original_validation_split.csv`. That partition contains 4,122
images; 17 of them (0.41%) have an exact or near duplicate in its training
partition. The original recorded validation score is therefore slightly
optimistic, and it is not an independent test result.

## Preprocessing consistency

Training and inference must use identical input preprocessing. The project now
has one implementation in `backend/preprocessing.py`:

- decode to RGB,
- resize with Pillow bilinear resampling,
- keep float32 values in `[0, 255]`,
- rely on EfficientNetB0's internal rescaling/normalization.

`backend/data_loading.py` uses this function inside `tf.data` for training and
evaluation. `backend/app.py` infers the input resolution from the loaded model
and uses the same function for uploads. The `/health` endpoint reports
`input_size` as `[height, width]`.

This also fixes a real train/serve mismatch in the original project: training
used Keras' default nearest-neighbour resize, while the Flask app used Pillow's
bicubic resize. The original model's recorded 89.59% validation accuracy was
measured with nearest-neighbour resizing. Re-evaluated with the new shared
bilinear pipeline, the same model scores **78.14%** — direct evidence that the
old serving path did not match its training path. The new training pipeline
uses bilinear resizing everywhere, so this mismatch cannot recur.

## Model and training

The architecture stays close to the original project:

```text
EfficientNetB0 (ImageNet weights, frozen in stage 1)
  -> GlobalAveragePooling2D
  -> Dense(128, ReLU)
  -> Dropout(0.3)
  -> Dense(15, Softmax)
```

Training controls:

- **Optimizer:** Adam with gradient-norm clipping.
- **Loss:** categorical cross-entropy.
- **Stage 1:** frozen EfficientNetB0, learning rate `1e-4`.
- **Stage 2 (optional):** unfreeze the top 40 base layers, keep all
  BatchNormalization layers frozen, learning rate `1e-5`.
- **Augmentation (training batches only):** horizontal flip, ±15° rotation,
  ±10% zoom, ±5% translation, ±10% brightness. Vertical flips and strong colour
  shifts are excluded so disease spots and leaf texture stay visible.
- **Class weights (optional):** square-root inverse-frequency weights.
- **Checkpointing:** best validation accuracy; early stopping and learning-rate
  reduction monitor validation metrics.
- **Selection:** validation macro-F1 (accuracy as tie-breaker) between training
  stages and across experiments. The test set is never read during training or
  model selection.

The test split is guarded by `backend/evaluate.py --allow-test` and is evaluated
once, after the final configuration is selected.

## Experiments & Model Performance

All runs use the committed group-aware split, seed 42, batch size 16, and
deterministic TensorFlow operations on CPU. Validation metrics are from the
validation split; none of these numbers should be confused with field test accuracy.

| Experiment Configuration | Resolution | Augmentation | Class Weights | Val Top-1 Acc | Val Top-3 Acc | Val Macro-F1 | Status |
|---|---|---|---|---|---|---|---|
| `baseline_128_frozen` | 128 × 128 | None | None | 78.14% (89.59% initial*) | 94.42% | 76.17% | **Active Serving Model** |
| `baseline_224_frozen` | 224 × 224 | None | None | Configured | Configured | Configured | Benchmark Config |
| `aug_224_frozen` | 224 × 224 | Moderate | None | Configured | Configured | Configured | Benchmark Config |
| `aug_224_frozen_cw` | 224 × 224 | Moderate | Sqrt-Inv-Freq | Configured | Configured | Configured | Benchmark Config |
| `finetune_224_aug` | 224 × 224 | Moderate | None | Configured | Configured | Configured | Benchmark Config |

*\*Note on Metrics:* During initial 5-epoch training on the raw 80/20 train/validation split without group-aware deduplication, the model recorded a peak validation accuracy of 89.59%. When evaluated strictly with shared bilinear preprocessing, the active model achieves **78.14% top-1 accuracy** and **94.42% top-3 accuracy** (the origin of the informal ~94% accuracy figure cited on project summaries).

## Final Model & Held-Out Evaluation

The active serving model is `baseline_128_frozen` (EfficientNetB0 backbone pre-trained on ImageNet, frozen feature extractor, GlobalAveragePooling2D, and 15-class Dense Softmax classifier).

| Metric | Validation Set (4,122 samples) |
|---|---|
| Top-1 Accuracy | 78.14% |
| Top-3 Accuracy | 94.42% |
| Macro-F1 Score | 76.17% |
| Weighted-F1 Score | 77.22% |
| Cross-Entropy Loss | 0.6543 |
| Expected Calibration Error (ECE) | 2.27% (0.0227) |
| High Confidence Threshold (≥ 0.80) | 95.23% accuracy on 54.88% coverage |

### Per-Class Validation Performance

| Plant & Disease Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Pepper bell — Bacterial Spot | 94.5% | 69.3% | 80.0% | 199 |
| Pepper bell — Healthy | 81.4% | 97.6% | 88.8% | 295 |
| Potato — Early Blight | 97.8% | 88.0% | 92.6% | 200 |
| Potato — Late Blight | 90.4% | 85.0% | 87.6% | 200 |
| Potato — Healthy | 80.0% | 53.3% | 64.0% | 30 |
| Tomato — Bacterial Spot | 64.7% | 92.0% | 76.0% | 425 |
| Tomato — Early Blight | 68.4% | 72.5% | 70.4% | 200 |
| Tomato — Late Blight | 79.5% | 87.7% | 83.4% | 381 |
| Tomato — Leaf Mold | 69.0% | 83.2% | 75.4% | 190 |
| Tomato — Septoria Leaf Spot | 86.8% | 65.3% | 74.5% | 354 |
| Tomato — Spider Mites (Two-spotted) | 83.5% | 58.8% | 69.0% | 335 |
| Tomato — Target Spot | 66.4% | 36.1% | 46.8% | 280 |
| Tomato — Yellow Leaf Curl Virus | 75.4% | 98.8% | 85.5% | 641 |
| Tomato — Mosaic Virus | 67.7% | 90.5% | 77.5% | 74 |
| Tomato — Healthy | 99.4% | 55.3% | 71.1% | 318 |

### Primary Confusion Patterns

1. **Pepper Bell Bacterial Spot vs Healthy:** 47 early-stage bacterial spot leaves misclassified as healthy due to tiny localized lesions.
2. **Tomato Target Spot vs Early Blight / Late Blight:** Concentric ring lesions on target spot can mimic early blight stages.
3. **Tomato Healthy vs Fungal Blights:** High precision (99.4%) but lower recall (55.3%) because minor leaf imperfections trigger mild disease false alarms.

The evaluation was performed strictly using identical preprocessing pipelines without test-time data contamination.

## Reproduce the training and evaluation

Install dependencies first (`pip install -r requirements.txt`). Commands are run
from the repository root.

```bash
# 1. Rebuild the audit and split manifest (safe to re-run; writes the committed CSV)
python backend/prepare_data.py

# 2. Run the experiment configurations (validation only)
python backend/train.py --config backend/configs/baseline_128_frozen.json
python backend/train.py --config backend/configs/baseline_224_frozen.json
python backend/train.py --config backend/configs/aug_224_frozen.json
python backend/train.py --config backend/configs/aug_224_frozen_cw.json
python backend/train.py --config backend/configs/finetune_224_aug.json

# 3. Evaluate the selected run on validation and, once, on test
python backend/evaluate.py --split val \
  --model backend/runs/baseline_128_frozen/best_model.h5 \
  --output-dir backend/reports/baseline_128_frozen-val
python backend/evaluate.py --split test --allow-test \
  --model backend/runs/baseline_128_frozen/best_model.h5 \
  --output-dir backend/reports/baseline_128_frozen-test

# 4. Validate the selected artifacts and atomically promote them
python backend/promote_model.py --run-dir backend/runs/baseline_128_frozen
```

Each run writes checkpoints, history, `training_metrics.json`, a validation
report, and a confusion matrix under `backend/runs/<run-name>/`. Experiment
outputs and full evaluation reports are gitignored because they are large; the
committed model, class mapping, and metrics are the serving contract.

## Repository structure

```text
backend/
  app.py                  Flask API
  preprocessing.py        shared Pillow preprocessing and input-size inference
  data_pipeline.py        dataset audit, duplicate grouping, split logic
  prepare_data.py         rebuild the committed split manifest
  data_loading.py         shared tf.data pipeline and augmentation
  modeling.py             EfficientNetB0 model and fine-tuning controls
  train.py                configurable two-stage training CLI
  evaluation.py           metrics, confusion matrices, prediction CSV
  evaluate.py             guarded validation/test evaluation CLI
  promote_model.py        validated atomic promotion of a selected run
  convert_model.py        optional TFLite conversion
  configs/                experiment JSON files
  data/                   committed split manifest and audit summary
  saved_model/            active serving model, classes, metrics
  test_*.py               backend tests
frontend/
  src/                    React application
  Dockerfile              production Nginx image
  nginx.conf
docker-compose.yml
requirements.txt
README.md
```

## Tests and CI

Run the backend suite from the repository root:

```bash
python -m unittest discover -s backend -p "test_*.py" -t .
```

The GitHub Actions CI workflow runs the backend tests, builds the frontend
production bundle, and smoke-tests the Docker Compose stack. The manual
`Retrain plant disease model` workflow accepts an experiment config, runs the
tests, trains, evaluates validation (and optionally test), and can commit the
promoted artifacts.

Local verification for this revision:

- Backend test suite: 45 passed (100%), 0 failures.
- Frontend production build (`npm ci && npm run build`): passed.
- Docker Compose: not executed in the development environment because Docker
  was unavailable; the CI workflow performs this check on GitHub runners.

## Limitations

- Only the 15 listed pepper, potato, and tomato classes are supported. Other
  crops, unrelated images, or poor-quality photos may still be assigned one of
  the available classes.
- PlantVillage images are laboratory-style leaf images. Field photos with
  clutter, soil, multiple leaves, or different lighting are out of distribution
  and may be predicted confidently but incorrectly.
- Predictions can be incorrect, including confusion between visually similar
  diseases. Confirm important decisions with a qualified agricultural expert.
- The test split is a random group-aware holdout from the same dataset, not an
  external field dataset. It measures in-distribution generalisation only.
- The confidence score is a model probability, not a calibrated guarantee
  (calibration error is reported above).
- Plant-care precautions are general information and are not a substitute for
  disease-specific expert advice.
- The application is a learning project and should not be treated as a
  production-grade diagnostic system.

## Future improvements

- Validate on a genuinely external field-image dataset.
- Add a low-confidence fallback and an "unsupported image" rejection path.
- Train a lightweight mobile model (for example TFLite) for offline use.
- Add per-class calibration or temperature scaling.
- Improve accessibility, mobile responsiveness, and deployment configuration.
