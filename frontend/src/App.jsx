import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState('');
  const [precautionsVisible, setPrecautionsVisible] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPrediction('');
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData);
      const { prediction } = response.data;
      setPrediction(prediction);
      setPrecautionsVisible(false);

      if (voiceEnabled) {
        speakPrediction(prediction);
      }
    } catch (error) {
      setPrediction('Prediction failed');
      console.error('Prediction error:', error);
    }
  };

  const speakPrediction = (text) => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
  };

  const handlePrecautions = () => {
    setPrecautionsVisible(true);
  };

  return (
    <div className="app">
      <div className="container">
        <h1>🌿 Plant Leaf Disease Detector</h1>
        <input type="file" onChange={handleImageChange} />
        {previewUrl && <img src={previewUrl} alt="preview" className="preview" />}
        <div className="button-group">
          <button onClick={handlePredict}>Predict</button>
          <button onClick={handlePrecautions}>Show Precautions</button>
          <label>
            <input type="checkbox" checked={voiceEnabled} onChange={() => setVoiceEnabled(!voiceEnabled)} />
            Voice Output
          </label>
        </div>
        <p><strong>Disease:</strong> {prediction}</p>

        {precautionsVisible && (
          <div className="precautions">
            <h3>🌱 General Plant Care Precautions:</h3>
            <ul>
              <li>Avoid overwatering the plant.</li>
              <li>Ensure proper sunlight and ventilation.</li>
              <li>Use disease-free seeds and tools.</li>
              <li>Apply appropriate fungicide or pesticide if necessary.</li>
              <li>Remove and destroy infected leaves or plants.</li>
              <li>Maintain regular monitoring and hygiene.</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
