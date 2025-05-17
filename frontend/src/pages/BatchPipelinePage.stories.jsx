import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import MultiPipelineSelector from '../components/MultiPipelineSelector';
import FileUpload from '../components/FileUpload';

// This is a mockup component we're creating just for Storybook testing
const BatchPipelinePage = () => {
  const [fileLogId, setFileLogId] = React.useState(null);
  const [uploadSuccess, setUploadSuccess] = React.useState(false);
  const [uploadMessage, setUploadMessage] = React.useState('');
  const [batchRuns, setBatchRuns] = React.useState([]);
  const [isLoading, setIsLoading] = React.useState(false);
  
  const handleUpload = async (file) => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 800)); // Simulate API call
    setFileLogId('mock-file-' + Date.now().toString().slice(-6));
    setUploadSuccess(true);
    setUploadMessage('File uploaded successfully!');
    setIsLoading(false);
  };
  
  const handleStartBatch = (selectedPipelines) => {
    setIsLoading(true);
    setTimeout(() => {
      // Simulate batch creation
      const newBatch = {
        batchId: 'batch-' + Date.now().toString().slice(-6),
        fileLogId,
        status: 'PROCESSING',
        pipelines: selectedPipelines.map(type => ({
          pipelineType: type,
          runId: 'run-' + type.toLowerCase() + '-' + Date.now().toString().slice(-4),
          status: 'QUEUED',
          result: null
        }))
      };
      setBatchRuns(prev => [newBatch, ...prev]);
      setIsLoading(false);
    }, 1000);
  };
  
  return (
    <div className="container">
      <h2 className="mb-4">Multi-Pipeline Executor</h2>
      
      <div className="card mb-4">
        <h3 className="mb-3">1. Upload Document</h3>
        <FileUpload 
          onUpload={handleUpload} 
          accept="application/pdf" 
          disabled={isLoading} 
        />
        
        {uploadMessage && (
          <div className={`alert ${uploadSuccess ? 'alert-success' : 'alert-danger'} mt-3`}>
            {uploadMessage}
            {uploadSuccess && fileLogId && <span> (File ID: {fileLogId})</span>}
          </div>
        )}
      </div>
      
      {uploadSuccess && fileLogId && (
        <div className="mb-4">
          <MultiPipelineSelector
            fileLogId={fileLogId}
            onSubmit={handleStartBatch}
            disabled={isLoading}
          />
        </div>
      )}
      
      {batchRuns.length > 0 && (
        <div className="card">
          <h3 className="mb-3">Batch Pipeline Runs</h3>
          
          {batchRuns.map(batch => (
            <div key={batch.batchId} className="mb-4">
              <div className="info-box">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <h4 className="mb-0">Batch #{batch.batchId}</h4>
                  <span className="badge bg-info">Status: {batch.status}</span>
                </div>
                <p className="mb-2">File ID: {batch.fileLogId}</p>
                <p className="mb-3">Running {batch.pipelines.length} pipeline(s)</p>
                
                <div className="result-container">
                  <table className="table table-bordered">
                    <thead>
                      <tr>
                        <th>Pipeline</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batch.pipelines.map(pipeline => (
                        <tr key={pipeline.runId}>
                          <td>{pipeline.pipelineType}</td>
                          <td>{pipeline.status}</td>
                          <td>
                            <button className="btn btn-sm btn-primary">View Details</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default {
  title: 'Pages/BatchPipelinePage',
  component: BatchPipelinePage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'A page for running multiple pipelines on a document in a batch.'
      }
    }
  },
  decorators: [
    (Story) => (
      <BrowserRouter>
        <div style={{ padding: '20px' }}>
          <Story />
        </div>
      </BrowserRouter>
    ),
  ],
};

// Default state
export const Default = {}; 
 