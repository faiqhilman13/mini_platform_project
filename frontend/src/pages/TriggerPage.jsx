import React, { useState } from 'react';
import { triggerPipeline } from '../services/api';

function TriggerPage() {
  const [fileLogId, setFileLogId] = useState('');
  const [pipelineType, setPipelineType] = useState('PDF_SUMMARIZER'); // Default to PDF_SUMMARIZER
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!fileLogId) {
      setError('Please enter a File Log ID.');
      return;
    }
    setIsLoading(true);
    setResponse(null);
    setError('');
    try {
      const result = await triggerPipeline(fileLogId, pipelineType);
      setResponse(result);
    } catch (err) {
      setError(err.message || 'Failed to trigger pipeline. See console for details.');
      setResponse(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: 'auto', padding: 20 }}>
      <h2>Trigger Pipeline</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 15 }}>
          <label htmlFor="fileLogId" style={{ display: 'block', marginBottom: 5 }}>File Log ID:</label>
          <input
            type="number"
            id="fileLogId"
            value={fileLogId}
            onChange={(e) => setFileLogId(e.target.value)}
            placeholder="Enter File Log ID from upload"
            required
            style={{ width: '100%', padding: 8, boxSizing: 'border-box' }}
          />
        </div>
        <div style={{ marginBottom: 15 }}>
          <label htmlFor="pipelineType" style={{ display: 'block', marginBottom: 5 }}>Pipeline Type:</label>
          <select 
            id="pipelineType" 
            value={pipelineType} 
            onChange={(e) => setPipelineType(e.target.value)}
            style={{ width: '100%', padding: 8, boxSizing: 'border-box' }}
          >
            <option value="PDF_SUMMARIZER">PDF Summarizer</option>
            <option value="RAG_CHATBOT">RAG Chatbot</option>
            <option value="TEXT_CLASSIFIER">Text Classifier</option>
          </select>
        </div>
        <button 
          type="submit" 
          disabled={isLoading} 
          style={{
            padding: '10px 15px', 
            cursor: isLoading ? 'not-allowed' : 'pointer',
            backgroundColor: isLoading ? '#ccc' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '4px'
          }}
        >
          {isLoading ? 'Triggering...' : 'Trigger Pipeline'}
        </button>
      </form>

      {error && (
        <div style={{ color: 'red', marginTop: 20, padding: 10, border: '1px solid red', borderRadius: 4 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div style={{ marginTop: 20, padding: 10, border: '1px solid #eee', borderRadius: 4, backgroundColor: '#f8f9fa' }}>
          <h3>Pipeline Triggered Successfully:</h3>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', color: '#333' }}>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default TriggerPage;
