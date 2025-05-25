import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';

/**
 * A component for selecting multiple pipelines to run on a document.
 * 
 * @param {Object} props Component props
 * @param {string} props.fileLogId ID of the uploaded file
 * @param {Function} props.onSubmit Callback when pipelines are submitted (receives array of selected pipeline types)
 * @param {boolean} [props.disabled=false] Whether the component is disabled
 * @param {string[]} [props.preSelectedPipelines=[]] Pre-selected pipeline types
 * @returns {JSX.Element} The MultiPipelineSelector component
 */
function MultiPipelineSelector({ 
  fileLogId, 
  onSubmit,
  disabled = false,
  preSelectedPipelines = [] 
}) {
  const [selectedPipelines, setSelectedPipelines] = useState(preSelectedPipelines);
  
  // Update selected pipelines when preSelectedPipelines changes
  useEffect(() => {
    setSelectedPipelines(preSelectedPipelines);
  }, [preSelectedPipelines]);
  
  const handleTogglePipeline = (pipelineType) => {
    if (selectedPipelines.includes(pipelineType)) {
      setSelectedPipelines(prev => prev.filter(p => p !== pipelineType));
    } else {
      setSelectedPipelines(prev => [...prev, pipelineType]);
    }
  };
  
  const pipelineOptions = [
    {
      id: 'PDF_SUMMARIZER',
      name: 'PDF Summarizer',
      description: 'Generate a concise summary of your document'
    },
    {
      id: 'RAG_CHATBOT',
      name: 'RAG Chatbot',
      description: 'Create a question-answering assistant from your document'
    },
    {
      id: 'TEXT_CLASSIFIER',
      name: 'Text Classifier',
      description: 'Categorize your document content'
    }
  ];
  
  return (
    <div className="card">
      <h3 className="mb-3">Select Pipelines to Run</h3>
      <p className="mb-3">
        You can select multiple pipelines to run on your document {fileLogId && <span>(File ID: {fileLogId})</span>}
      </p>
      
      <div className="form-group">
        {pipelineOptions.map(pipeline => (
          <div key={pipeline.id} className="mb-2">
            <label className="d-flex align-items-center">
              <input 
                type="checkbox" 
                checked={selectedPipelines.includes(pipeline.id)}
                onChange={() => handleTogglePipeline(pipeline.id)}
                disabled={disabled}
                className="mr-2"
              />
              <span><strong>{pipeline.name}</strong> - {pipeline.description}</span>
            </label>
          </div>
        ))}
        
        <button 
          onClick={() => onSubmit(selectedPipelines)}
          disabled={disabled || selectedPipelines.length === 0}
          className="form-control mt-3"
        >
          Run Selected Pipelines ({selectedPipelines.length})
        </button>
      </div>
    </div>
  );
}

MultiPipelineSelector.propTypes = {
  fileLogId: PropTypes.string.isRequired,
  onSubmit: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  preSelectedPipelines: PropTypes.arrayOf(PropTypes.string)
};

export default MultiPipelineSelector; 