import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction('');
    setConfidence('');
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData);
      setPrediction(response.data.label);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error(error);
      setPrediction('Error: Unable to classify image.');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Lung Cancer Classification</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Analyzing...' : 'Predict'}
      </button>

      {prediction && (
        <div className="result">
          <h2>Prediction: {prediction}</h2>
          <p>Confidence: {confidence}</p>
        </div>
      )}
    </div>
  );
}

export default App;
