// frontend/src/components/ImageUpload.js
import React from 'react';
import '../styles/ImageUpload.css';

const ImageUpload = ({ onImageSelect, onProcess, isLoading }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    onImageSelect(file);
  };

  return (
    <div className="image-upload-container">
      <h3>1. Upload Image</h3>
      <input
        type="file"
        accept="image/png, image/jpeg, image/jpg"
        onChange={handleFileChange}
      />
      <button
        className="process-button"
        onClick={onProcess}
        disabled={isLoading}
      >
        {isLoading ? 'Processing...' : '2. Process Image'}
      </button>
    </div>
  );
};

export default ImageUpload;
