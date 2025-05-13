export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:8000/api/v1/upload/', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMsg = 'Upload failed.';
    try {
      // Try to parse the error response from the backend
      const errData = await response.json();
      errorMsg = errData.detail || errData.message || errorMsg;
      if (Array.isArray(errData.detail) && errData.detail.length > 0) { // Handle FastAPI validation errors
        errorMsg = errData.detail.map(d => `${d.loc.join(' -> ')} - ${d.msg}`).join('; ');
      }
    } catch (e) {
      // If parsing fails, use the response text or a generic message
      errorMsg = response.statusText || errorMsg;
    }
    throw new Error(errorMsg);
  }

  return await response.json();
}

export async function triggerPipeline(fileLogId, pipelineType) {
  const payload = {
    uploaded_file_log_id: parseInt(fileLogId, 10),
    pipeline_type: pipelineType,
  };

  const response = await fetch('http://127.0.0.1:8000/api/v1/pipelines/trigger', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMsg = 'Failed to trigger pipeline.';
    try {
      const errData = await response.json();
      errorMsg = errData.detail || errData.message || errorMsg;
      if (Array.isArray(errData.detail) && errData.detail.length > 0) {
        errorMsg = errData.detail.map(d => `${d.loc.join(' -> ')} - ${d.msg}`).join('; ');
      }
    } catch (e) {
      errorMsg = response.statusText || errorMsg;
    }
    throw new Error(errorMsg);
  }
  return await response.json();
}

export async function getPipelineStatus(runId) {
  const response = await fetch(`http://127.0.0.1:8000/api/v1/pipelines/${runId}/status`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    let errorMsg = 'Failed to fetch pipeline status.';
    try {
      const errData = await response.json();
      errorMsg = errData.detail || errData.message || errorMsg;
      if (Array.isArray(errData.detail) && errData.detail.length > 0) {
        errorMsg = errData.detail.map(d => `${d.loc.join(' -> ')} - ${d.msg}`).join('; ');
      }
    } catch (e) {
      errorMsg = response.statusText || errorMsg;
    }
    throw new Error(errorMsg);
  }
  return await response.json();
}

export async function sendRagQuestion(runId, question) {
  const payload = {
    question: question,
    pipeline_run_id: runId
  };

  const response = await fetch(`http://127.0.0.1:8000/api/v1/rag/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMsg = 'Failed to get answer from RAG chatbot.';
    try {
      const errData = await response.json();
      errorMsg = errData.detail || errData.message || errorMsg;
      if (Array.isArray(errData.detail) && errData.detail.length > 0) {
        errorMsg = errData.detail.map(d => `${d.loc.join(' -> ')} - ${d.msg}`).join('; ');
      }
    } catch (e) {
      errorMsg = response.statusText || errorMsg;
    }
    throw new Error(errorMsg);
  }
  return await response.json();
}

// Add other API functions here as needed, for example:
// export async function getPipelineStatus(runId) { ... }
