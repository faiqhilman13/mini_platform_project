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
    <div className="container">
      <h2 className="mb-4">Upload a PDF File</h2>
      <FileUpload onUpload={handleUpload} accept="application/pdf" disabled={isLoading} />
      
      {uploadMessage && (
        <div className={`alert ${uploadSuccess ? 'alert-success' : 'alert-danger'} mt-3`}>
          {uploadMessage}
          {uploadSuccess && fileLogId && <span> (File Log ID: {fileLogId})</span>}
        </div>
      )}

      {uploadSuccess && fileLogId && (
        <div className="card mt-4">
          <h3 className="mb-3">Select Pipeline</h3>
          <div className="form-group">
            <select 
              className="form-control mb-3"
              value={selectedPipeline} 
              onChange={(e) => setSelectedPipeline(e.target.value)} 
              disabled={isLoading}
            >
              <option value={PipelineType.PDF_SUMMARIZER}>PDF Summarizer</option>
              <option value={PipelineType.RAG_CHATBOT}>RAG Chatbot</option>
              <option value={PipelineType.TEXT_CLASSIFIER}>Text Classifier</option>
            </select>
            <button 
              onClick={handleTriggerPipeline} 
              disabled={isLoading || !fileLogId} 
              className="form-control"
            >
              {isLoading ? (
                <span><div className="loading"></div> Processing...</span>
              ) : (
                'Trigger Pipeline'
              )}
            </button>
          </div>
        </div>
      )}

      {pipelineMessage && (
        <div className={`alert ${pipelineRun && pipelineRun.status === 'FAILED' ? 'alert-danger' : 'alert-info'} mt-3`}>
          {pipelineMessage}
        </div>
      )}

      {pipelineRun && pipelineRun.status === 'COMPLETED' && (
        <div className="result-container">
          <h4 className="mb-3">Pipeline Result:</h4>
          {selectedPipeline === PipelineType.PDF_SUMMARIZER && (
            <pre>
              {pipelineRun.result ? JSON.stringify(pipelineRun.result, null, 2) : 'No result data.'}
            </pre>
          )}
          {selectedPipeline === PipelineType.TEXT_CLASSIFIER && (
             <p className="mb-0"><strong>Category:</strong> {pipelineRun.result ? pipelineRun.result.category : 'N/A'}</p>
          )}
          {/* RAG Chatbot interface will be handled differently, likely a new component */} 
        </div>
      )}
    </div>
  );
}

export default UploadPage;
