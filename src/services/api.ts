import axios from 'axios';
import { 
  UploadedFile, 
  PipelineRun, 
  MLPipelineRun,
  MLModel,
  DatasetProfileSummary,
  ChatMessage
} from '../types';

// Set base URL from environment variable or use default for development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// File upload API
export const uploadFile = async (file: File): Promise<UploadedFile> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const getUploadedFiles = async (): Promise<UploadedFile[]> => {
  const response = await api.get('/upload/files');
  return response.data;
};

export const deleteFile = async (fileId: string): Promise<{ message: string; success: boolean }> => {
  const response = await api.delete(`/upload/files/${fileId}`);
  return response.data;
};

// Pipeline API
export const triggerPipeline = async (
  fileId: string, 
  pipelineType: string, 
  config?: Record<string, any>
): Promise<PipelineRun> => {
  try {
    // Try multiple formats to work around backend issues
    const requestBody = {
      uploaded_file_log_id: parseInt(fileId),
      pipeline_type: pipelineType,
      config: {
        ...config,
        // Try multiple ways to specify the target column
        target_column: config?.target_variable || config?.target_column,
        target: config?.target_variable || config?.target_column,
        targetColumn: config?.target_variable || config?.target_column,
      },
      // Also try at the root level
      target_variable: config?.target_variable,
      target_column: config?.target_variable,
      target: config?.target_variable,
    };
    
    console.log('Triggering pipeline with multiple target formats:', requestBody);
    
    const response = await api.post('/pipelines/trigger', requestBody);
    
    console.log('Pipeline trigger response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error triggering pipeline:', error);
    if (axios.isAxiosError(error)) {
      console.error('Response status:', error.response?.status);
      console.error('Response data:', error.response?.data);
      console.error('Request URL:', error.config?.url);
    }
    throw error;
  }
};

export const getPipelineStatus = async (runUuid: string): Promise<PipelineRun> => {  const response = await api.get(`/pipelines/${runUuid}/status`);  return response.data;};

export const getPipelineRuns = async (fileId?: string): Promise<PipelineRun[]> => {
  const url = fileId ? `/pipelines/runs?file_id=${fileId}` : '/pipelines/runs';
  const response = await api.get(url);
  return response.data;
};

// RAG Chatbot API
export const sendChatMessage = async (
  runUuid: string, 
  message: string
): Promise<ChatMessage> => {
  const response = await api.post(`/rag/ask`, {
    pipeline_run_id: runUuid,
    question: message,
  });
  
  return {
    id: crypto.randomUUID(),
    content: response.data.answer,
    role: 'assistant',
    timestamp: new Date().toISOString(),
  };
};

export const getChatHistory = async (runUuid: string): Promise<ChatMessage[]> => {
  // Note: RAG system doesn't store chat history - it's stateless
  // This function returns empty array for now
  // Chat history should be stored locally in the browser
  return [];
};

// Dataset API
export const getDatasetPreview = async (fileId: string, rows: number = 10): Promise<any[]> => {
  try {
    console.log(`Making API call to: ${API_BASE_URL}/data/${fileId}/preview?rows=${rows}`);
    const response = await api.get(`/data/${fileId}/preview?rows=${rows}`);
    console.log('Dataset preview API response:', response.data);
    
    // Handle both possible response structures
    if (response.data.sample_rows) {
      return response.data.sample_rows;
    } else if (response.data.preview?.sample_rows) {
      return response.data.preview.sample_rows;
    } else {
      console.warn('Sample rows not found in expected structure, returning response data');
      return response.data;
    }
  } catch (error) {
    console.error('Error fetching dataset preview:', error);
    if (axios.isAxiosError(error)) {
      console.error('Response status:', error.response?.status);
      console.error('Response data:', error.response?.data);
      console.error('Request URL:', error.config?.url);
    }
    throw error;
  }
};

export const getDatasetProfile = async (fileId: string): Promise<DatasetProfileSummary> => {
  try {
    console.log(`Making API call to: ${API_BASE_URL}/data/${fileId}/profile`);
    const response = await api.get(`/data/${fileId}/profile`);
    console.log('Dataset profile API response:', response.data);
    
    // Extract the profile data from the nested structure
    if (response.data.profile) {
      console.log('Extracted profile data:', response.data.profile);
      return response.data.profile;
    } else {
      console.log('Profile data not found in expected structure');
      return response.data;
    }
  } catch (error) {
    console.error('Error fetching dataset profile:', error);
    if (axios.isAxiosError(error)) {
      console.error('Response status:', error.response?.status);
      console.error('Response data:', error.response?.data);
      console.error('Request URL:', error.config?.url);
    }
    throw error;
  }
};

// ML Pipeline API
export const triggerMLPipeline = async (
  fileId: string,
  config: Record<string, any>
): Promise<MLPipelineRun> => {
  const response = await api.post('/pipelines/ml/trigger', {
    uploaded_file_log_id: parseInt(fileId),
    target_variable: config.target_variable || config.targetColumn,
    problem_type: config.problem_type || config.problemType,
    algorithms: config.algorithms || [],
    preprocessing_config: config.preprocessing || config.preprocessing_config || {},
    experiment_name: config.experiment_name,
    experiment_description: config.experiment_description,
  });
  
  return response.data;
};

export const getMLPipelineStatus = async (runUuid: string): Promise<MLPipelineRun> => {
  const response = await api.get(`/pipelines/ml/status/${runUuid}`);
  return response.data;
};

export const getMLModels = async (runUuid: string): Promise<MLModel[]> => {
  const response = await api.get(`/pipelines/ml/models/${runUuid}`);
  return response.data;
};

export const getMLModel = async (modelId: string): Promise<MLModel> => {
  const response = await api.get(`/pipelines/ml/model/${modelId}`);
  return response.data;
};

// Algorithm API
export const getAlgorithmSuggestions = async (problemType?: string): Promise<any[]> => {
  const url = problemType 
    ? `/algorithms/suggestions?problem_type=${problemType}`
    : '/algorithms/suggestions';
  const response = await api.get(url);
  return response.data;
};

export const getSupportedAlgorithms = async (problemType?: string): Promise<any[]> => {
  const url = problemType 
    ? `/algorithms/supported?problem_type=${problemType}`
    : '/algorithms/supported';
  const response = await api.get(url);
  return response.data;
};

export const validateMLConfiguration = async (config: Record<string, any>): Promise<any> => {
  const response = await api.post('/pipelines/ml/validate', config);
  return response.data;
};

export default api;