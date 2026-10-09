import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

// Backend base URL, configurable at build time via a Vite env variable.
// Falls back to the Flask development server for local runs.
// Example: create frontend/.env with VITE_API_URL=http://127.0.0.1:5000
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_FILE_BYTES = 8 * 1024 * 1024; // Must stay within the backend's MAX_UPLOAD_MB.

function App() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState('');
  const [isPredicting, setIsPredicting] = useState(false);
  const [precautionsVisible, setPrecautionsVisible] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  // Revoke the preview object URL when it is replaced or on unmount.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleImageChange = (e) => {
    const files = e.target.files;
    // The dialog was cancelled or the selection cleared: keep current state.
    if (!files || files.length === 0) {
      return;
    }
    const file = files[0];

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError('Please choose a JPEG, PNG, WebP, or BMP image.');
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setError('Image is too large. Please choose a file under 8 MB.');
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setImage(file);
    setPrediction('');
    setConfidence(null);
    setError('');
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image || isPredicting) return;

    const formData = new FormData();
    formData.append('file', image);

    setIsPredicting(true);
    setError('');
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData);
      const { prediction: label, confidence: score } = response.data;
      setPrediction(label);
      setConfidence(score);
      setPrecautionsVisible(false);

      if (voiceEnabled) {
        speakPrediction(label);
      }
    } catch (err) {
      const message =
        err?.response?.data?.error ||
        'Prediction failed. Is the backend running at ' + API_BASE_URL + '?';
      setError(message);
      setPrediction('');
      setConfidence(null);
      console.error('Prediction error:', err);
    } finally {
      setIsPredicting(false);
    }
  };

  const speakPrediction = (text) => {
    if (!('speechSynthesis' in window)) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  const handlePrecautions = () => {
    setPrecautionsVisible(true);
  };

  return (
    <div className="app">
      <div className="container">
        <h1>🌿 Plant Leaf Disease Detector</h1>
        <input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={handleImageChange} />
        {previewUrl && <img src={previewUrl} alt="preview" className="preview" />}
        <div className="button-group">
          <button onClick={handlePredict} disabled={!image || isPredicting}>
            {isPredicting ? 'Predicting…' : 'Predict'}
          </button>
          <button onClick={handlePrecautions}>Show Precautions</button>
          <label>
            <input type="checkbox" checked={voiceEnabled} onChange={() => setVoiceEnabled(!voiceEnabled)} />
            Voice Output
          </label>
        </div>
        {error && <p className="error">{error}</p>}
        {prediction && (
          <p>
            <strong>Disease:</strong> {prediction}
            {confidence !== null && (
              <> (<strong>Confidence:</strong> {(confidence * 100).toFixed(1)}%)</>
            )}
          </p>
        )}
        {prediction && (
          <p className="disclaimer">
            Confidence is the model's self-reported score, not a guarantee that
            the prediction is correct. Confirm critical diagnoses independently.
          </p>
        )}

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
