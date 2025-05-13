import React, { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import { uploadFile, triggerPipeline, getPipelineStatus } from '../services/api';

// Matches the backend enum app/models/pipeline_models.py
const PipelineType = {
  PDF_SUMMARIZER: 'PDF_SUMMARIZER',
  RAG_CHATBOT: 'RAG_CHATBOT',
  TEXT_CLASSIFIER: 'TEXT_CLASSIFIER',
};

function UploadPage() {
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [fileLogId, setFileLogId] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [selectedPipeline, setSelectedPipeline] = useState(PipelineType.PDF_SUMMARIZER);
  const [pipelineRun, setPipelineRun] = useState(null);
  const [pipelineMessage, setPipelineMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pollingIntervalId, setPollingIntervalId] = useState(null);

  const handleUpload = async (file) => {
    setUploadSuccess(false);
    setFileLogId(null);
    setUploadMessage('');
    setPipelineRun(null);
    setPipelineMessage('');
    setIsLoading(true);
    try {
      const response = await uploadFile(file);
      setUploadSuccess(true);
      setFileLogId(response.file_log_id);
      setUploadMessage(response.message || 'File uploaded successfully!');
    } catch (err) {
      setUploadSuccess(false);
      setUploadMessage(err.message || 'Upload failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTriggerPipeline = async () => {
    if (!fileLogId) {
      setPipelineMessage('Please upload a file first.');
      return;
    }
    setIsLoading(true);
    setPipelineRun(null);
    setPipelineMessage('');
    try {
      const response = await triggerPipeline(fileLogId, selectedPipeline);
      setPipelineRun(response);
      setPipelineMessage(`Pipeline ${selectedPipeline} triggered. Run UUID: ${response.run_uuid}`);
      // Start polling for status if it's a summarizer or classifier
      if (selectedPipeline === PipelineType.PDF_SUMMARIZER || selectedPipeline === PipelineType.TEXT_CLASSIFIER) {
        if (pollingIntervalId) clearInterval(pollingIntervalId); // Clear existing interval
        const intervalId = setInterval(() => pollStatus(response.run_uuid), 5000);
        setPollingIntervalId(intervalId);
      }
    } catch (err) {
      setPipelineMessage(err.message || 'Failed to trigger pipeline.');
    } finally {
      setIsLoading(false);
    }
  };

  const pollStatus = async (runUuid) => {
    try {
      const statusResponse = await getPipelineStatus(runUuid);
      setPipelineRun(prevRun => ({ ...prevRun, ...statusResponse }));
      if (statusResponse.status === 'COMPLETED' || statusResponse.status === 'FAILED') {
        if (pollingIntervalId) clearInterval(pollingIntervalId);
        setPollingIntervalId(null);
        setPipelineMessage(prev => `${prev} - Status: ${statusResponse.status}. ${statusResponse.result ? 'Result available.' : ''} ${statusResponse.error_message || ''}`);
      }
    } catch (error) {
      console.error('Failed to poll status:', error);
      setPipelineMessage(prev => `${prev} - Error polling status.`);
      if (pollingIntervalId) clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
  };

  // Cleanup interval on component unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [pollingIntervalId]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 40, padding: 20, maxWidth: 600, margin: '40px auto', border: '1px solid #ccc', borderRadius: 8 }}>
      <h2>Upload a PDF File</h2>
      <FileUpload onUpload={handleUpload} accept="application/pdf" disabled={isLoading} />
      {uploadMessage && (
        <div style={{ color: uploadSuccess ? 'green' : 'red', marginTop: 16 }}>
          {uploadMessage}
          {uploadSuccess && fileLogId && <span> (File Log ID: {fileLogId})</span>}
        </div>
      )}

      {uploadSuccess && fileLogId && (
        <div style={{ marginTop: 20, borderTop: '1px solid #eee', paddingTop: 20, width: '100%' }}>
          <h3>Select Pipeline</h3>
          <select 
            value={selectedPipeline} 
            onChange={(e) => setSelectedPipeline(e.target.value)} 
            disabled={isLoading}
            style={{ padding: 8, marginRight: 10, minWidth: 200 }}
          >
            <option value={PipelineType.PDF_SUMMARIZER}>PDF Summarizer</option>
            <option value={PipelineType.RAG_CHATBOT}>RAG Chatbot</option>
            <option value={PipelineType.TEXT_CLASSIFIER}>Text Classifier</option>
          </select>
          <button onClick={handleTriggerPipeline} disabled={isLoading || !fileLogId} style={{ padding: '8px 16px' }}>
            {isLoading ? 'Processing...' : 'Trigger Pipeline'}
          </button>
        </div>
      )}

      {pipelineMessage && (
        <div style={{ marginTop: 16, color: pipelineRun && pipelineRun.status === 'FAILED' ? 'red' : 'blue' }}>
          {pipelineMessage}
        </div>
      )}

      {pipelineRun && pipelineRun.status === 'COMPLETED' && (
        <div style={{ marginTop: 20, borderTop: '1px solid #eee', paddingTop: 20, width: '100%' }}>
          <h4>Pipeline Result:</h4>
          {selectedPipeline === PipelineType.PDF_SUMMARIZER && (
            <pre style={{ whiteSpace: 'pre-wrap', backgroundColor: '#f5f5f5', padding: 10, borderRadius: 4 }}>
              {pipelineRun.result ? JSON.stringify(pipelineRun.result, null, 2) : 'No result data.'}
            </pre>
          )}
          {selectedPipeline === PipelineType.TEXT_CLASSIFIER && (
             <p><strong>Category:</strong> {pipelineRun.result ? pipelineRun.result.category : 'N/A'}</p>
          )}
          {/* RAG Chatbot interface will be handled differently, likely a new component */} 
        </div>
      )}
    </div>
  );
}

export default UploadPage;
