import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, Workflow, Play, FileType } from 'lucide-react';
import PageLayout from '../components/layout/PageLayout';
import PipelineSelector from '../components/pipeline/PipelineSelector';
import PipelineStatus from '../components/pipeline/PipelineStatus';
import Button from '../components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import Spinner from '../components/ui/Spinner';
import { UploadedFile, PipelineType, PipelineRun, FileType as FileTypeEnum } from '../types';
import { getUploadedFiles, getPipelineRuns, triggerPipeline, getPipelineStatus } from '../services/api';
import { formatBytes, formatDate, isDatasetFile } from '../utils/helpers';

const FileDetailsPage = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  
  const [file, setFile] = useState<UploadedFile | null>(null);
  const [fileLoading, setFileLoading] = useState(true);
  const [fileError, setFileError] = useState<string | null>(null);
  
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRun[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState<string | null>(null);
  
  const [selectedPipeline, setSelectedPipeline] = useState<PipelineType | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Fetch file details
  const fetchFile = useCallback(async () => {
    if (!fileId) return;
    
    try {
      setFileLoading(true);
      const files = await getUploadedFiles();
      const foundFile = files.find(f => f.id === fileId);
      
      if (foundFile) {
        setFile(foundFile);
        setFileError(null);
      } else {
        setFileError('File not found');
      }
    } catch (err) {
      setFileError('Failed to load file details');
      console.error('Error fetching file:', err);
    } finally {
      setFileLoading(false);
    }
  }, [fileId]);

  // Fetch pipeline runs for this file
  const fetchPipelineRuns = useCallback(async () => {
    if (!fileId) return;
    
    try {
      setRunsLoading(true);
      const runs = await getPipelineRuns(fileId);
      setPipelineRuns(runs);
      setRunsError(null);
    } catch (err) {
      setRunsError('Failed to load pipeline runs');
      console.error('Error fetching pipeline runs:', err);
    } finally {
      setRunsLoading(false);
    }
  }, [fileId]);

  // Update a specific pipeline run status
  const updatePipelineRunStatus = useCallback(async (runUuid: string) => {
    try {
      const updatedRun = await getPipelineStatus(runUuid);
      setPipelineRuns(prev => 
        prev.map(run => run.run_uuid === runUuid ? updatedRun : run)
      );
      
      // If status changed to COMPLETED, fetch all runs to get results
      if (updatedRun.status === 'COMPLETED' || updatedRun.status === 'FAILED') {
        fetchPipelineRuns();
      }
    } catch (err) {
      console.error('Error updating pipeline status:', err);
    }
  }, [fetchPipelineRuns]);

  useEffect(() => {
    fetchFile();
    fetchPipelineRuns();
  }, [fetchFile, fetchPipelineRuns]);

  // Handle pipeline selection
  const handlePipelineSelect = (pipelineType: PipelineType) => {
    setSelectedPipeline(pipelineType);
    setLaunchError(null);
  };

  // Launch the selected pipeline
  const handleLaunchPipeline = async () => {
    if (!fileId || !selectedPipeline) return;
    
    try {
      setIsLaunching(true);
      setLaunchError(null);
      
      const createResponse = await triggerPipeline(fileId, selectedPipeline);
      
      // Fetch the complete pipeline run details
      const pipelineRun = await getPipelineStatus(createResponse.run_uuid);
      setPipelineRuns(prev => [pipelineRun, ...prev]);
      setSelectedPipeline(null);
      
      // If this is a ML_TRAINING pipeline, navigate to the ML config page
      if (selectedPipeline === 'ML_TRAINING') {
        navigate(`/ml/${pipelineRun.run_uuid}/config`);
      }
      // If this is a RAG_CHATBOT pipeline, navigate to the chat page
      else if (selectedPipeline === 'RAG_CHATBOT') {
        navigate(`/chat/${pipelineRun.run_uuid}`);
      }
    } catch (err) {
      setLaunchError('Failed to launch pipeline');
      console.error('Error launching pipeline:', err);
    } finally {
      setIsLaunching(false);
    }
  };

  // Handle refresh for an active pipeline
  const handleRefreshPipeline = (runUuid: string) => {
    updatePipelineRunStatus(runUuid);
  };

  // Handle clicking on a completed pipeline to view results
  const handlePipelineClick = (pipelineRun: PipelineRun) => {
    if (pipelineRun.status !== 'COMPLETED') return;

    switch (pipelineRun.pipeline_type) {
      case 'PDF_SUMMARIZER':
        return 'Summarizer';
      case 'RAG_CHATBOT':
        return 'RAG Chatbot';
      case 'ML_TRAINING':
        return 'ML Training';
      default:
        return pipelineRun.pipeline_type;
    }
  };

  // Render loading state
  if (fileLoading) {
    return (
      <PageLayout>
        <div className="flex justify-center items-center py-12">
          <Spinner size="lg" />
        </div>
      </PageLayout>
    );
  }

  // Render error state
  if (fileError || !file) {
    return (
      <PageLayout>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
          <p className="text-gray-300 mb-6">{fileError || 'File not found'}</p>
          <Button onClick={() => navigate('/')}>Back to Home</Button>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="space-y-6">
        <Button
          variant="ghost"
          size="sm"
          icon={<ChevronLeft className="w-4 h-4" />}
          onClick={() => navigate('/')}
        >
          Back to Files
        </Button>

        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">{file.filename}</h1>
            <div className="flex items-center text-sm text-gray-500">
              <FileType className="w-4 h-4 mr-1" />
              <span>{file.file_type.toUpperCase()}</span>
              <span className="mx-2">•</span>
              <span>{formatBytes(file.size_bytes)}</span>
              <span className="mx-2">•</span>
              <span>Uploaded on {formatDate(file.upload_timestamp)}</span>
            </div>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Workflow className="h-5 w-5 mr-2 text-blue-600" />
              Select Pipeline
            </CardTitle>
          </CardHeader>
          <CardContent>
            <PipelineSelector
              fileType={file.file_type as FileTypeEnum}
              onSelect={handlePipelineSelect}
              className="mb-6"
            />
            
            {selectedPipeline && (
              <div className="flex justify-end">
                <Button
                  onClick={handleLaunchPipeline}
                  disabled={isLaunching}
                  isLoading={isLaunching}
                  icon={<Play className="w-4 h-4" />}
                >
                  Launch Pipeline
                </Button>
              </div>
            )}
            
            {launchError && (
              <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                {launchError}
              </div>
            )}
          </CardContent>
        </Card>

        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Pipeline Runs</h2>
          
          {runsLoading ? (
            <div className="flex justify-center items-center py-8">
              <Spinner size="md" />
            </div>
          ) : runsError ? (
            <div className="text-center py-8 text-red-600">
              {runsError}
            </div>
          ) : pipelineRuns.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No pipeline runs yet. Select a pipeline above to get started.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                          {pipelineRuns.map(run => (                <PipelineStatus                  key={run.run_uuid}                  pipelineRun={run}                  onRefresh={() => handleRefreshPipeline(run.run_uuid)}                  onClick={handlePipelineClick}                />              ))}
            </div>
          )}
        </div>
      </div>
    </PageLayout>
  );
};

export default FileDetailsPage;
