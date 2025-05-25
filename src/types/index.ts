export type FileType = 'pdf' | 'csv' | 'xlsx' | 'text' | 'json' | 'unknown';

export interface UploadedFile {
  id: string;
  filename: string;
  file_type: FileType;
  size_bytes: number;
  upload_timestamp: string;
  content_type?: string;
  message?: string;
}

export type PipelineType = 'PDF_SUMMARIZER' | 'RAG_CHATBOT' | 'TEXT_CLASSIFIER' | 'ML_TRAINING';

export type PipelineStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface PipelineRun {
  run_uuid: string;
  uploaded_file_log_id: string;
  pipeline_type: PipelineType;
  status: PipelineStatus;
  result?: any;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface MLPipelineRun extends PipelineRun {
  problem_type: 'CLASSIFICATION' | 'REGRESSION';
  target_variable: string;
  algorithms_config: Record<string, any>;
  preprocessing_config: Record<string, any>;
  metrics?: Record<string, any>;
  best_model_id?: string;
}

export interface MLModel {
  model_id: string;
  pipeline_run_id: string;
  algorithm_name: string;
  hyperparameters: Record<string, any>;
  performance_metrics: Record<string, any>;
  model_path: string;
  feature_importance?: Record<string, number>;
  training_time: number;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
}

export interface DatasetColumn {
  name: string;
  data_type: string;
  missing_count: number;
  unique_count: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  mode?: string | number;
  std?: number;
  sample_values: string[];
}

export interface DatasetProfileSummary {
  file_id: string;
  total_rows: number;
  total_columns: number;
  missing_cells: number;
  duplicate_rows: number;
  memory_usage: string;
  columns: DatasetColumn[];
}

export interface AlgorithmOption {
  name: string;
  displayName: string;
  description: string;
  hyperparameters: HyperParameter[];
  problemTypes: ('CLASSIFICATION' | 'REGRESSION')[];
}

export interface HyperParameter {
  name: string;
  displayName: string;
  type: 'number' | 'boolean' | 'select' | 'string';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description: string;
}

export interface Algorithm {
  name: string;
  display_name: string;
  description: string;
  problem_types: string[];
  hyperparameters: AlgorithmHyperParameter[];
  default_metrics: string[];
  recommended_preprocessing: string[];
  min_samples: number;
  supports_feature_importance: boolean;
  supports_probabilities: boolean;
  training_complexity: 'low' | 'medium' | 'high';
}

export interface AlgorithmHyperParameter {
  name: string;
  type: string;
  default: any;
  min_value?: number;
  max_value?: number;
  allowed_values?: any[];
  description: string;
  required: boolean;
}

export interface AlgorithmConfig {
  algorithm_name: string;
  hyperparameters: Record<string, any>;
  is_enabled: boolean;
}

export interface AlgorithmSuggestion {
  name: string;
  display_name: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  recommended_for: string[];
  pros: string[];
  cons: string[];
}

export interface PreprocessingConfig {
  targetColumn: string;
  problemType: 'CLASSIFICATION' | 'REGRESSION';
  trainTestSplit: number;
  missingValueStrategy: 'mean' | 'median' | 'mode' | 'constant' | 'drop';
  constantValue?: string;
  scaling: 'none' | 'standard' | 'minmax';
  categoricalEncoding: 'onehot' | 'label';
  featureSelection: string[];
}