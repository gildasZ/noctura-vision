// frontend/src/components/ControlPanel.js
import React from 'react';
import ModelSelector from './ModelSelector';
import '../styles/ControlPanel.css';

const ControlPanel = ({
  models,
  selectedModel,
  setSelectedModel,
  applyEnhancement,
  setApplyEnhancement,
  scoreThreshold,
  setScoreThreshold,
  inputMaxDim,
  setInputMaxDim
}) => {
  return (
    <div className="controls-panel">
      <h2>Controls</h2>
      <div className="control-item">
        <label htmlFor="model-select">Segmentation Model</label>
        <ModelSelector
          models={models}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
        />
      </div>

      <div className="control-item">
        <label htmlFor="enhancement-toggle" className="checkbox-label">
          <input
            type="checkbox"
            id="enhancement-toggle"
            checked={applyEnhancement}
            onChange={(e) => setApplyEnhancement(e.target.checked)}
          />
          Apply Low-Light Enhancement
        </label>
      </div>

      <div className="control-item">
        <label htmlFor="threshold-slider">Confidence Threshold: {scoreThreshold}</label>
        <input
          type="range"
          id="threshold-slider"
          min="0.1"
          max="0.9"
          step="0.05"
          value={scoreThreshold}
          onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-item">
        <label htmlFor="resolution-select">Processing Resolution</label>
        <select
          id="resolution-select"
          value={inputMaxDim}
          onChange={(e) => setInputMaxDim(parseInt(e.target.value, 10))}
        >
          <option value="480">480px (Fast)</option>
          <option value="720">720px (Balanced)</option>
          <option value="1080">1080px (Quality)</option>
        </select>
      </div>
    </div>
  );
};

export default ControlPanel;
