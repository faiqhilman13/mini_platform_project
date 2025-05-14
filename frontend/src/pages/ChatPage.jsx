import React, { useState, useEffect, useRef } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { getPipelineStatus, sendRagQuestion } from '../services/api';

function ChatPage() {
  const { runId } = useParams();
  const location = useLocation();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [pipelineData, setPipelineData] = useState(null);
  const messagesEndRef = useRef(null);

  // Load pipeline data on component mount
  useEffect(() => {
    if (runId) {
      fetchPipelineStatus();
    }
  }, [runId]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchPipelineStatus = async () => {
    setIsLoading(true);
    setError('');
    try {
      const result = await getPipelineStatus(runId);
      setPipelineData(result);
      // Add a welcome message if this is a new chat
      if (messages.length === 0) {
        setMessages([
          {
            role: 'system',
            content: `Welcome! This is a RAG-powered chatbot for the document "${result.result?.doc_id || 'Unknown'}". Ask questions about the document content.`
          }
        ]);
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch RAG pipeline status.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !runId) return;

    const userMessage = { role: 'user', content: input };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Add a new API function to send a RAG question
      const response = await sendRagQuestion(runId, input);
      
      const botMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources
      };
      
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (err) {
      setError(err.message || 'Failed to get response from the RAG chatbot.');
      const errorMessage = {
        role: 'system',
        content: `Error: ${err.message || 'Failed to get response from the RAG chatbot.'}`,
        isError: true
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!runId) {
    return (
      <div style={{ maxWidth: 800, margin: 'auto', padding: 20 }}>
        <h2>RAG Chatbot</h2>
        <div style={{ color: 'red', marginTop: 20 }}>
          <strong>Error:</strong> No RAG pipeline run ID provided. Please trigger a RAG pipeline first.
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 800, margin: 'auto', padding: 20 }}>
      <h2>RAG Chatbot</h2>
      {pipelineData && (
        <div 
          className="info-box"
          style={{ 
            marginBottom: 20, 
            padding: 10, 
            border: '1px solid #eee', 
            borderRadius: 4, 
            backgroundColor: '#f8f9fa',
          }}>
          <h3>Document Information</h3>
          <p><strong>Pipeline ID:</strong> {runId}</p>
          <p><strong>Status:</strong> {pipelineData.status}</p>
          <p><strong>Document:</strong> {pipelineData.result?.doc_id || 'Unknown'}</p>
          <p><strong>Chunks:</strong> {pipelineData.result?.chunks || 'Unknown'}</p>
        </div>
      )}

      {pipelineData && pipelineData.status !== 'COMPLETED' && (
        <div style={{ color: 'orange', marginBottom: 20 }}>
          <strong>Note:</strong> This pipeline is not yet completed (status: {pipelineData.status}). 
          You may not be able to chat until processing is complete.
        </div>
      )}

      <div style={{ 
        height: '400px', 
        border: '1px solid #ddd', 
        borderRadius: '4px', 
        padding: '10px', 
        marginBottom: '10px',
        overflow: 'auto',
        backgroundColor: '#fff' 
      }}>
        {messages.map((msg, index) => (
          <div 
            key={index} 
            style={{ 
              marginBottom: '10px',
              textAlign: msg.role === 'user' ? 'right' : 'left'
            }}
          >
            <div style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: '12px',
              maxWidth: '80%',
              backgroundColor: msg.isError ? '#ffcccc' : 
                              msg.role === 'user' ? '#e3f2fd' : 
                              msg.role === 'system' ? '#f5f5f5' : '#e8f5e9',
              textAlign: 'left',
              color: msg.isError ? '#a00' : '#333'
            }}>
              <p style={{ margin: '0', wordBreak: 'break-word' }}>{msg.content}</p>
              
              {msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: '8px', fontSize: '0.8em', borderTop: '1px solid #eee', paddingTop: '4px' }}>
                  <strong>Sources:</strong>
                  <ul style={{ margin: '4px 0 0 0', paddingLeft: '20px' }}>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        {source.title || source.source} {source.page && `(Page ${source.page})`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the document..."
          disabled={isLoading || (pipelineData && pipelineData.status !== 'COMPLETED')}
          style={{
            flex: 1,
            padding: '10px',
            borderRadius: '4px 0 0 4px',
            border: '1px solid #ddd',
            borderRight: 'none'
          }}
        />
        <button 
          type="submit" 
          disabled={isLoading || !input.trim() || (pipelineData && pipelineData.status !== 'COMPLETED')}
          style={{
            padding: '10px 15px',
            backgroundColor: isLoading ? '#ccc' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '0 4px 4px 0',
            cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer'
          }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>

      {error && (
        <div style={{ color: 'red', marginTop: 10 }}>
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
}

export default ChatPage; 