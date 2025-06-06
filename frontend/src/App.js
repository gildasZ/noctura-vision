// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './styles/App.css';

import ControlPanel from './components/ControlPanel';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';

// Backend API URL
const API_URL = "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState('image'); // 'image' or 'video'

  // State for all controls
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [applyEnhancement, setApplyEnhancement] = useState(true);
  const [scoreThreshold, setScoreThreshold] = useState(0.5);
  const [inputMaxDim, setInputMaxDim] = useState(1080);

  // State for image processing flow
  const [originalImageFile, setOriginalImageFile] = useState(null);
  const [originalImageURL, setOriginalImageURL] = useState('');
  const [processedImageURL, setProcessedImageURL] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch available models from the backend when the app loads
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/v1/models`);
        setModels(response.data.models);
        if (response.data.models.length > 0) {
          setSelectedModel(response.data.models[0].value); // Set default model
        }
      } catch (err) {
        console.error("Failed to fetch models:", err);
        setError("Could not connect to the backend to fetch models. Is it running?");
      }
    };
    fetchModels();
  }, []);

  // Handler for when a new image is selected
  const handleImageSelect = (file) => {
    setOriginalImageFile(file);
    setProcessedImageURL(''); // Clear previous result
    setError('');
    if (file) {
      setOriginalImageURL(URL.createObjectURL(file));
    } else {
      setOriginalImageURL('');
    }
  };

  // Handler for the "Process Image" button click
  const handleProcessImage = async () => {
    if (!originalImageFile) {
      setError("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    setError('');
    setProcessedImageURL('');

    const formData = new FormData();
    formData.append('file', originalImageFile);
    formData.append('model_type', selectedModel);
    formData.append('apply_enhancement', applyEnhancement);
    formData.append('score_threshold', scoreThreshold);
    formData.append('input_max_dim', inputMaxDim);

    try {
      const response = await axios.post(`${API_URL}/api/v1/process_image`, formData, {
        responseType: 'blob', // Important: expect a binary response
      });
      const imageBlob = new Blob([response.data], { type: 'image/jpeg' });
      const imageUrl = URL.createObjectURL(imageBlob);
      setProcessedImageURL(imageUrl);
    } catch (err) {
      console.error("Error processing image:", err);
      const errText = err.response ? (await err.response.data.text()) : "An unknown error occurred.";
      setError(`Processing failed. Server says: ${errText}`);
    } finally {
      setIsLoading(false);
    }
  };


  const renderContent = () => {
    if (mode === 'image') {
      return (
        <div className="main-content">
          <ImageUpload onImageSelect={handleImageSelect} onProcess={handleProcessImage} isLoading={isLoading} />
          <ResultDisplay
            originalImageURL={originalImageURL}
            processedImageURL={processedImageURL}
            isLoading={isLoading}
            error={error}
          />
        </div>
      );
    } else {
      return (
        <div className="main-content">
          {/* Placeholders for Video Mode */}
          <div className="placeholder-component">Original Webcam Feed</div>
          <div className="placeholder-component">Processed Video Stream</div>
        </div>
      );
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🦉 NocturaVision</h1>
        <p>Perceive the Night. Understand the Invisible.</p>
      </header>

      <nav className="App-nav">
        <button onClick={() => setMode('image')} className={mode === 'image' ? 'active' : ''}>
          Image Processing
        </button>
        <button onClick={() => setMode('video')} className={mode === 'video' ? 'active' : ''}>
          Live Video
        </button>
      </nav>

      <main className="App-main">
        <ControlPanel
          models={models}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          applyEnhancement={applyEnhancement}
          setApplyEnhancement={setApplyEnhancement}
          scoreThreshold={scoreThreshold}
          setScoreThreshold={setScoreThreshold}
          inputMaxDim={inputMaxDim}
          setInputMaxDim={setInputMaxDim}
        />
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
