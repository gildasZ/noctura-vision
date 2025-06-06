// frontend/src/components/ResultDisplay.js
import React from 'react';
import '../styles/ResultDisplay.css';

const ResultDisplay = ({ originalImageURL, processedImageURL, isLoading, error }) => {
  const renderContent = () => {
    if (isLoading) {
      return <div className="loader"></div>;
    }
    if (error) {
      return <div className="error-message">Error: {error}</div>;
    }
    if (!originalImageURL) {
      return <p>Upload an image and click "Process Image" to see the results here.</p>;
    }
    return (
      <>
        <div className="image-container">
          <h4>Original</h4>
          <img src={originalImageURL} alt="Original" />
        </div>
        <div className="image-container">
          <h4>Processed</h4>
          {processedImageURL ? (
            <img src={processedImageURL} alt="Processed" />
          ) : (
            <div className="placeholder">
              <p>Result will appear here</p>
            </div>
          )}
        </div>
      </>
    );
  };

  return (
    <div className="result-display-container">
      {renderContent()}
    </div>
  );
};

export default ResultDisplay;
