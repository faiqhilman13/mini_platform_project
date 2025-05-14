import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getPipelineStatus } from '../services/api';

function StatusPage() {
  const { runId: urlRunId } = useParams();
  const navigate = useNavigate();
  const [runId, setRunId] = useState(urlRunId || '');
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineData, setPipelineData] = useState(null);
  const [error, setError] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchStatus = useCallback(async (currentRunId) => {
    if (!currentRunId) return;
    setIsLoading(true);
    setError('');
    try {
      const result = await getPipelineStatus(currentRunId);
      setPipelineData(result);
    } catch (err) {
      setError(err.message || 'Failed to fetch status. See console for details.');
      setPipelineData(null);
    }
    setIsLoading(false);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!runId) {
      setError('Please enter a Run ID.');
      setPipelineData(null);
      return;
    }
    fetchStatus(runId);
  };

  const goToChat = () => {
    if (pipelineData && pipelineData.pipeline_type === 'RAG_CHATBOT' && pipelineData.status === 'COMPLETED') {
      navigate(`/chat/${pipelineData.run_uuid}`);
    }
  };

  useEffect(() => {
    if (urlRunId) {
      fetchStatus(urlRunId);
    }
  }, [urlRunId, fetchStatus]);

  useEffect(() => {
    let intervalId;
    if (autoRefresh && runId && pipelineData && 
        (pipelineData.status === 'PENDING' || pipelineData.status === 'RUNNING')) {
      intervalId = setInterval(() => {
        fetchStatus(runId);
      }, 5000); // Refresh every 5 seconds
    }
    return () => clearInterval(intervalId);
  }, [autoRefresh, runId, pipelineData, fetchStatus]);

  // Helper function to render pipeline results based on type
  const renderPipelineResults = () => {
    if (!pipelineData || !pipelineData.result) return null;
    
    switch (pipelineData.pipeline_type) {
      case 'PDF_SUMMARIZER':
        return (
          <div style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #ccc' }}>
            <h4>Summary:</h4>
            {pipelineData.result.summary && Array.isArray(pipelineData.result.summary) ? (
              <ul style={{ fontFamily: 'monospace', backgroundColor: '#e9ecef', padding: 15, borderRadius: 4, color: '#333' }}>
                {pipelineData.result.summary.map((point, idx) => (
                  <li key={idx} style={{ marginBottom: 8 }}>{point}</li>
                ))}
              </ul>
            ) : (
              <p style={{ fontFamily: 'monospace', backgroundColor: '#e9ecef', padding: 10, borderRadius: 4, color: '#333' }}>
                {JSON.stringify(pipelineData.result, null, 2)}
              </p>
            )}
          </div>
        );
        
      case 'TEXT_CLASSIFIER':
        return (
          <div style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #ccc' }}>
            <h4>Classification Result:</h4>
            <div style={{ padding: 15, backgroundColor: '#e9ecef', borderRadius: 4 }}>
              <p><strong>Category:</strong> 
                <span style={{ 
                  display: 'inline-block', 
                  marginLeft: 10, 
                  padding: '4px 12px', 
                  borderRadius: 12, 
                  backgroundColor: '#28a745', 
                  color: 'white',
                  fontWeight: 'bold' 
                }}>
                  {pipelineData.result.category}
                </span>
              </p>
              {pipelineData.result.confidence && (
                <p><strong>Confidence:</strong> {(pipelineData.result.confidence * 100).toFixed(2)}%</p>
              )}
              {pipelineData.result.keywords && (
                <div>
                  <p><strong>Keywords:</strong></p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {pipelineData.result.keywords.map((keyword, idx) => (
                      <span key={idx} style={{ 
                        backgroundColor: '#007bff', 
                        color: 'white', 
                        padding: '2px 8px', 
                        borderRadius: 12,
                        fontSize: '0.9em'
                      }}>
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {pipelineData.result.explanation && (
                <div style={{ marginTop: 10 }}>
                  <p><strong>Explanation:</strong></p>
                  <p style={{ fontStyle: 'italic' }}>{pipelineData.result.explanation}</p>
                </div>
              )}
            </div>
          </div>
        );
        
      case 'RAG_CHATBOT':
        return (
          <div style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #ccc' }}>
            <h4>RAG Document Ingestion:</h4>
            <div style={{ padding: 15, backgroundColor: '#e9ecef', borderRadius: 4 }}>
              <p><strong>Document ID:</strong> {pipelineData.result.doc_id}</p>
              <p><strong>Chunks Created:</strong> {pipelineData.result.chunks}</p>
              <p><strong>Message:</strong> {pipelineData.result.message}</p>
              <div style={{ marginTop: 15 }}>
                <button
                  onClick={goToChat}
                  style={{
                    padding: '8px 15px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Chat with this Document
                </button>
              </div>
            </div>
          </div>
        );
        
      default:
        return (
          <div style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #ccc' }}>
            <h4>Pipeline Result:</h4>
            <pre style={{ whiteSpace: 'pre-wrap', overflowX: 'auto', backgroundColor: '#e9ecef', padding: 10, borderRadius: 4 }}>
              {JSON.stringify(pipelineData.result, null, 2)}
            </pre>
          </div>
        );
    }
  };

  return (
    <div style={{ maxWidth: 700, margin: 'auto', padding: 20 }}>
      <h2>Check Pipeline Status</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: 20 }}>
        <div style={{ marginBottom: 10 }}>
          <label htmlFor="runId" style={{ display: 'block', marginBottom: 5 }}>Run ID (UUID):</label>
          <input
            type="text"
            id="runId"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            placeholder="Enter Run ID (UUID) from trigger step"
            required
            style={{ width: 'calc(100% - 110px)', padding: 8, boxSizing: 'border-box', marginRight: 10 }}
          />
          <button 
            type="submit" 
            disabled={isLoading || !runId}
            style={{
              padding: '8px 15px', 
              cursor: (isLoading || !runId) ? 'not-allowed' : 'pointer',
              backgroundColor: (isLoading || !runId) ? '#ccc' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px'
            }}
          >
            {isLoading ? 'Fetching...' : 'Get Status'}
          </button>
        </div>
        <div>
          <label htmlFor="autoRefreshCheckbox" style={{ cursor: 'pointer' }}>
            <input 
              type="checkbox" 
              id="autoRefreshCheckbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              disabled={!runId || (pipelineData && pipelineData.status !== 'PENDING' && pipelineData.status !== 'RUNNING')}
            />
            Auto-refresh status (if Pending/Running)
          </label>
        </div>
      </form>

      {error && (
        <div style={{ color: 'red', marginTop: 20, padding: 10, border: '1px solid red', borderRadius: 4 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {pipelineData && (
        <div 
          className="info-box"
          style={{
            marginTop: 20, 
            padding: 10, 
            border: '1px solid #eee', 
            borderRadius: 4, 
            backgroundColor: '#f8f9fa',
          }}>
          <h3>Pipeline Status Details:</h3>
          <div style={{ marginBottom: 15 }}>
            <p><strong>Run ID:</strong> {pipelineData.run_uuid}</p>
            <p><strong>Pipeline Type:</strong> {pipelineData.pipeline_type}</p>
            <p><strong>Status:</strong> 
              <span style={{ 
                display: 'inline-block', 
                marginLeft: 10, 
                padding: '2px 8px', 
                borderRadius: 12, 
                backgroundColor: 
                  pipelineData.status === 'COMPLETED' ? '#28a745' : 
                  pipelineData.status === 'FAILED' ? '#dc3545' : 
                  pipelineData.status === 'RUNNING' ? '#007bff' : 
                  '#ffc107', 
                color: 'white',
                fontSize: '0.9em' 
              }}>
                {pipelineData.status}
              </span>
            </p>
            <p><strong>Created:</strong> {new Date(pipelineData.created_at).toLocaleString()}</p>
            <p><strong>Last Updated:</strong> {new Date(pipelineData.updated_at).toLocaleString()}</p>
          </div>
          
          {pipelineData.status === 'COMPLETED' && renderPipelineResults()}
          
          {pipelineData.status === 'FAILED' && pipelineData.error_message && (
            <div style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #ccc' }}>
              <h4>Error Details:</h4>
              <p style={{ color: 'red', backgroundColor: '#ffebee', padding: 10, borderRadius: 4 }}>
                {pipelineData.error_message}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default StatusPage;
