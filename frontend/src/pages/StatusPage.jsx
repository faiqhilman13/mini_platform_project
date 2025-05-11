import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { getPipelineStatus } from '../services/api';

function StatusPage() {
  const { runId: urlRunId } = useParams();
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

  useEffect(() => {
    if (urlRunId) {
      fetchStatus(urlRunId);
    }
  }, [urlRunId, fetchStatus]);

  useEffect(() => {
    let intervalId;
    if (autoRefresh && runId && pipelineData && 
        (pipelineData.status === 'QUEUED' || pipelineData.status === 'PROCESSING')) {
      intervalId = setInterval(() => {
        fetchStatus(runId);
      }, 5000); // Refresh every 5 seconds
    }
    return () => clearInterval(intervalId);
  }, [autoRefresh, runId, pipelineData, fetchStatus]);

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
              disabled={!runId || (pipelineData && pipelineData.status !== 'QUEUED' && pipelineData.status !== 'PROCESSING')}
            />
            Auto-refresh status (if Queued/Processing)
          </label>
        </div>
      </form>

      {error && (
        <div style={{ color: 'red', marginTop: 20, padding: 10, border: '1px solid red', borderRadius: 4 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {pipelineData && (
        <div style={{ marginTop: 20, padding: 10, border: '1px solid #eee', borderRadius: 4, backgroundColor: '#f8f9fa' }}>
          <h3>Pipeline Status Details:</h3>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', color: '#333' }}>{JSON.stringify(pipelineData, null, 2)}</pre>
          {pipelineData.status === 'COMPLETED' && pipelineData.output_reference && (
            <div style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #ccc' }}>
              <h4>Summary Output:</h4>
              <p style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', backgroundColor: '#e9ecef', padding: 10, borderRadius: 4, color: '#333' }}>{pipelineData.output_reference}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default StatusPage;
