-- Mini IDP Database Schema - PostgreSQL Optimized
-- Initial migration for transitioning from SQLite to Supabase

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 1. File Upload Tracking Table
CREATE TABLE uploadedfilelog (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100),
    size_bytes BIGINT NOT NULL,
    storage_location TEXT NOT NULL,
    is_dataset BOOLEAN DEFAULT FALSE,
    file_uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Additional metadata for Supabase
    created_by UUID REFERENCES auth.users(id) DEFAULT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 2. Pipeline Run Tracking Table  
CREATE TABLE pipelinerun (
    id SERIAL PRIMARY KEY,
    run_uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    uploaded_file_log_id INTEGER REFERENCES uploadedfilelog(id) ON DELETE CASCADE,
    pipeline_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    error_message TEXT,
    result_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Performance and audit fields
    execution_time_ms INTEGER,
    created_by UUID REFERENCES auth.users(id) DEFAULT NULL
);

-- 3. Data Profiling Cache Table
CREATE TABLE dataprofiling (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES uploadedfilelog(id) ON DELETE CASCADE,
    profile_data JSONB DEFAULT '{}'::jsonb,
    column_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Summary fields for quick access
    total_rows INTEGER DEFAULT 0,
    total_columns INTEGER DEFAULT 0,
    quality_score REAL DEFAULT 1.0,
    has_target_suggestions BOOLEAN DEFAULT FALSE,
    
    -- Cache control
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '24 hours')
);

-- 4. ML Pipeline Run Table
CREATE TABLE ml_pipeline_run (
    id SERIAL PRIMARY KEY,
    run_uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    uploaded_file_log_id INTEGER REFERENCES uploadedfilelog(id) ON DELETE CASCADE,
    pipeline_type VARCHAR(50) DEFAULT 'ML_TRAINING',
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    
    -- ML Configuration
    problem_type VARCHAR(20) NOT NULL CHECK (problem_type IN ('REGRESSION', 'CLASSIFICATION')),
    target_variable VARCHAR(100) NOT NULL,
    feature_count INTEGER DEFAULT 0,
    ml_config JSONB DEFAULT '{}'::jsonb,
    algorithms_config JSONB DEFAULT '{}'::jsonb,
    preprocessing_config JSONB DEFAULT '{}'::jsonb,
    
    -- Results Summary
    total_models_trained INTEGER DEFAULT 0,
    best_model_id VARCHAR(100),
    best_model_score REAL,
    best_model_metric VARCHAR(50),
    
    -- Performance Metrics
    total_training_time_seconds REAL,
    dataset_rows_used INTEGER,
    dataset_features_used INTEGER,
    data_quality_score REAL,
    preprocessing_warnings JSONB DEFAULT '[]'::jsonb,
    
    -- Error Handling
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit
    created_by UUID REFERENCES auth.users(id) DEFAULT NULL
);

-- 5. ML Experiment Table (for grouping related runs)
CREATE TABLE ml_experiment (
    id SERIAL PRIMARY KEY,
    experiment_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Dataset Reference
    dataset_file_id INTEGER REFERENCES uploadedfilelog(id) ON DELETE CASCADE,
    problem_type VARCHAR(20) NOT NULL CHECK (problem_type IN ('REGRESSION', 'CLASSIFICATION')),
    target_variable VARCHAR(100) NOT NULL,
    
    -- Experiment Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    
    -- Summary Statistics (computed from runs)
    total_runs INTEGER DEFAULT 0,
    best_score REAL,
    best_algorithm VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Organization
    tags JSONB DEFAULT '[]'::jsonb,
    created_by UUID REFERENCES auth.users(id) DEFAULT NULL
);

-- 6. ML Model Storage Table
CREATE TABLE ml_model (
    id SERIAL PRIMARY KEY,
    model_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    pipeline_run_id INTEGER REFERENCES ml_pipeline_run(id) ON DELETE CASCADE,
    
    -- Model Information
    algorithm_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    
    -- Performance Metrics
    training_score REAL,
    validation_score REAL,
    test_score REAL,
    metrics JSONB DEFAULT '{}'::jsonb,
    feature_importance JSONB DEFAULT '{}'::jsonb,
    
    -- Model Storage
    model_path TEXT,
    model_size_bytes BIGINT,
    
    -- Performance Characteristics
    training_time_seconds REAL,
    prediction_time_ms REAL,
    model_metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Status
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_best_model BOOLEAN DEFAULT FALSE,
    is_deployed BOOLEAN DEFAULT FALSE
);

-- Performance Indexes
CREATE INDEX idx_uploadedfilelog_file_uuid ON uploadedfilelog(file_uuid);
CREATE INDEX idx_uploadedfilelog_timestamp ON uploadedfilelog(upload_timestamp DESC);
CREATE INDEX idx_uploadedfilelog_user ON uploadedfilelog(created_by);

CREATE INDEX idx_pipelinerun_run_uuid ON pipelinerun(run_uuid);
CREATE INDEX idx_pipelinerun_file_id ON pipelinerun(uploaded_file_log_id);
CREATE INDEX idx_pipelinerun_status ON pipelinerun(status);
CREATE INDEX idx_pipelinerun_timestamp ON pipelinerun(created_at DESC);

CREATE INDEX idx_dataprofiling_file_id ON dataprofiling(file_id);
CREATE INDEX idx_dataprofiling_expires ON dataprofiling(expires_at);

CREATE INDEX idx_ml_pipeline_run_uuid ON ml_pipeline_run(run_uuid);
CREATE INDEX idx_ml_pipeline_run_file_id ON ml_pipeline_run(uploaded_file_log_id);
CREATE INDEX idx_ml_pipeline_run_status ON ml_pipeline_run(status);

CREATE INDEX idx_ml_model_pipeline_run ON ml_model(pipeline_run_id);
CREATE INDEX idx_ml_model_best ON ml_model(is_best_model) WHERE is_best_model = true;

-- JSONB Indexes for better query performance
CREATE INDEX idx_pipelinerun_result_data ON pipelinerun USING GIN (result_data);
CREATE INDEX idx_ml_config_gin ON ml_pipeline_run USING GIN (ml_config);
CREATE INDEX idx_ml_metrics_gin ON ml_model USING GIN (metrics);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_pipelinerun_updated_at 
    BEFORE UPDATE ON pipelinerun 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dataprofiling_updated_at 
    BEFORE UPDATE ON dataprofiling 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_pipeline_run_updated_at 
    BEFORE UPDATE ON ml_pipeline_run 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_experiment_updated_at 
    BEFORE UPDATE ON ml_experiment 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) Setup (Optional - can be enabled later)
-- ALTER TABLE uploadedfilelog ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE pipelinerun ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE dataprofiling ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml_pipeline_run ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml_experiment ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml_model ENABLE ROW LEVEL SECURITY;

-- Basic RLS Policies (Optional - can be enabled later)
-- CREATE POLICY "Users can manage their own files" ON uploadedfilelog
--     FOR ALL USING (created_by = auth.uid());

-- CREATE POLICY "Users can manage their own pipeline runs" ON pipelinerun
--     FOR ALL USING (created_by = auth.uid());

-- Comments for documentation
COMMENT ON TABLE uploadedfilelog IS 'Tracks all uploaded files with metadata';
COMMENT ON TABLE pipelinerun IS 'Tracks pipeline execution status and results';
COMMENT ON TABLE dataprofiling IS 'Caches dataset profiling results with expiration';
COMMENT ON TABLE ml_pipeline_run IS 'Tracks ML training pipeline executions';
COMMENT ON TABLE ml_experiment IS 'Groups related ML runs for comparison';
COMMENT ON TABLE ml_model IS 'Stores trained model metadata and performance'; 