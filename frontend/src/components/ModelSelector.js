// frontend/src/components/ModelSelector.js
import React from 'react';

const ModelSelector = ({ models, selectedModel, setSelectedModel }) => {
  if (!models || models.length === 0) {
    return <select disabled><option>Loading models...</option></select>;
  }

  return (
    <select
      id="model-select"
      value={selectedModel}
      onChange={(e) => setSelectedModel(e.target.value)}
    >
      {models.map((model) => (
        <option key={model.value} value={model.value}>
          {model.label}
        </option>
      ))}
    </select>
  );
};

export default ModelSelector;
