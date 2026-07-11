# 🌿 Plant Leaf Disease Detection using Machine Learning

A full-stack web application to detect diseases in plant leaves using a **Convolutional Neural Network (CNN)** model. Users can upload a leaf image and receive disease predictions with confidence score and precautionary advice. Designed for accessibility and ease of use.

---

## 🧠 Overview

- **Goal**: Detect diseases in plant leaves using image-based ML model
- **Model**: CNN trained on the PlantVillage dataset
- **Frontend**: React (with Vite)
- **Backend**: Flask + TensorFlow
- **Voice Feature**: Speaks out the prediction (only for disease name)
- **Extra Modules Used**: `gTTS` for voice, `flask-cors` for CORS handling, `numpy`, `PIL`, etc.

---

## 📁 Folder Structure

```
plant-disease-app/
│
├── backend/
│   ├── app.py                         # Flask server with prediction route
│   ├── saved_model/
│   │   └── plant_disease_model.h5     # Trained CNN model
│
├── frontend/
│   ├── src/
│   │   └── App.jsx                    # React UI logic
│   ├── index.html
│
├── images/                            # Screenshots for README
│   ├── homepage.png
│   ├── uploading.png
│   ├── prediction_result.png
│   ├── precautions.png
│
├── requirements.txt                   # Backend Python dependencies
└── README.md
```

---

## ⚙️ How to Run the Project Locally

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/Chandini7203/plant-disease-app.git
cd plant-disease-app
```

### 🧪 2. Setup and Start the Backend

> Make sure you're in the root directory

```bash
pip install -r requirements.txt
cd backend
python app.py
```

### 🌐 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Now visit: [http://localhost:5173](http://localhost:5173)

---

## 📸 Screenshots

- **🏠 Homepage**  
  ![Homepage](./images/Homepage.png)

- **📤 Image Upload**  
  ![Uploading](./images/uploading.png)

- **📊 Prediction Result**  
  ![Prediction](./images/prediction_result.png)

- **💡 Precaution View**  
  ![Precautions](./images/precautions.png)

---

## ✅ Features

- 📷 Upload leaf image to predict disease  
- ⚡ Fast and accurate CNN predictions  
- 📢 Voice output using browser TTS (no external APIs)  
- 🌱 Precautions displayed for each disease  
- 💻 Simple, responsive, and user-friendly interface

---

## 📌 Technologies Used

| Component   | Tools/Frameworks             |
|-------------|-------------------------------|
| Frontend    | React (Vite), HTML, CSS, JS   |
| Backend     | Flask, Flask-CORS, Python     |
| Model       | TensorFlow, Keras, NumPy      |
| Utilities   | Pillow (PIL), SpeechSynthesis |
| Dataset     | [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🧠 Model Details

- **Architecture**: CNN with custom layers  
- **Input Size**: 128x128 pixels  
- **Activation**: Softmax on final layer  
- **Loss**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Epochs**: 5  
- **Accuracy Achieved**: ~94%

---

## 🚀 Git Commands to Push Changes

```bash
git add .
git commit -m "Updated files"
git push origin main
```

---

