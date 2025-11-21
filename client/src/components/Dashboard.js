/**
 * Main Dashboard Component
 * Handles image upload, displays results, and shows history
 */
import React, { useState, useEffect } from 'react';
import { segmentationAPI } from '../api';
import Navbar from './Navbar';

function Dashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');

  // Load history when component mounts
  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await segmentationAPI.getHistory();
      setHistory(data);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError('');

    try {
      const data = await segmentationAPI.uploadImage(selectedFile);
      setResult(data);
      
      // Reload history to show new result
      await loadHistory();
    } catch (err) {
      setError(err.response?.data?.detail || 'Segmentation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleHistoryClick = async (id) => {
    try {
      const data = await segmentationAPI.getSegmentation(id);
      setResult(data);
      setSelectedFile(null);
      setPreview(null);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error('Failed to load segmentation:', err);
    }
  };

  return (
    <div className="app">
      <Navbar />
      
      <div className="main-container">
        <h1 className="page-title">Image Segmentation</h1>

        {/* Upload Section */}
        <div className="upload-section">
          <div className="upload-icon">📤</div>
          <p className="upload-text">
            {selectedFile ? selectedFile.name : 'Upload an image to run segmentation'}
          </p>
          
          <input
            type="file"
            id="file-input"
            className="file-input"
            accept="image/*"
            onChange={handleFileSelect}
          />
          
          <label htmlFor="file-input">
            <button
              className="btn-upload"
              onClick={() => document.getElementById('file-input').click()}
              disabled={loading}
            >
              Choose Image
            </button>
          </label>

          {selectedFile && (
            <button
              className="btn-upload"
              onClick={handleUpload}
              disabled={loading}
              style={{ marginLeft: '1rem' }}
            >
              {loading ? 'Processing...' : 'Run Model'}
            </button>
          )}
        </div>

        {error && <div className="error-message" style={{ textAlign: 'center' }}>{error}</div>}

        {/* Loading State */}
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Running segmentation model...</p>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="results-container">
            <h2 className="result-title" style={{ textAlign: 'center', marginBottom: '2rem' }}>
              Results
            </h2>
            
            <div className="results-grid">
              <div className="result-card">
                <h3 className="result-title">Original Image</h3>
                <img
                  src={`data:image/png;base64,${result.original_image}`}
                  alt="Original"
                  className="result-image"
                />
              </div>

              <div className="result-card">
                <h3 className="result-title">Segmentation Mask</h3>
                <img
                  src={`data:image/png;base64,${result.segmentation_mask}`}
                  alt="Segmentation"
                  className="result-image"
                />
              </div>
            </div>
          </div>
        )}

        {/* History Section */}
        <div className="history-section">
          <h2 className="history-title">Previous Results</h2>
          
          {history.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📁</div>
              <p className="empty-text">No segmentations yet. Upload an image to get started!</p>
            </div>
          ) : (
            <div className="history-grid">
              {history.map((item) => (
                <div
                  key={item.id}
                  className="history-card"
                  onClick={() => handleHistoryClick(item.id)}
                >
                  <img
                    src={`data:image/png;base64,${item.thumbnail}`}
                    alt={item.filename}
                    className="history-thumbnail"
                  />
                  <div className="history-filename">{item.filename}</div>
                  <div className="history-date">
                    {new Date(item.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
