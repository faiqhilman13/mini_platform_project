import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, FileText, Brain, AlertCircle, Eye, Download } from 'lucide-react';
import PageLayout from '../components/layout/PageLayout';
import Button from '../components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import Spinner from '../components/ui/Spinner';
import { PipelineRun } from '../types';
import { getPipelineStatus } from '../services/api';
import { formatDate } from '../utils/helpers';

const PipelineResultsPage = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  
  const [pipelineRun, setPipelineRun] = useState<PipelineRun | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch pipeline run details
  useEffect(() => {
    if (!runId) return;
    
    const fetchPipelineRun = async () => {
      try {
        setIsLoading(true);
        const run = await getPipelineStatus(runId);
        setPipelineRun(run);
        setError(null);
      } catch (err) {
        setError('Failed to load pipeline details');
        console.error('Error fetching pipeline run:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchPipelineRun();
  }, [runId]);

  const renderPipelineResults = () => {
    if (!pipelineRun || !pipelineRun.result) {
      return (
        <div className="text-center py-8 text-gray-500">
          No results available for this pipeline.
        </div>
      );
    }

            switch (pipelineRun.pipeline_type) {      case 'PDF_SUMMARIZER':
        return (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <FileText className="h-5 w-5 mr-2 text-purple-600" />
                  Document Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose max-w-none">
                  {pipelineRun.result.summary && Array.isArray(pipelineRun.result.summary) ? (
                    <div className="space-y-3">
                      {pipelineRun.result.summary.map((sentence: string, index: number) => (
                        <p key={index} className="text-gray-700 leading-relaxed">
                          {sentence}
                        </p>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-700 leading-relaxed">
                      {pipelineRun.result.summary || pipelineRun.result.content || 'Summary not available'}
                    </p>
                  )}
                </div>
                
                {pipelineRun.result.metadata && (
                  <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Document Information</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {pipelineRun.result.metadata.pages && (
                        <div>
                          <span className="font-medium">Pages:</span> {pipelineRun.result.metadata.pages}
                        </div>
                      )}
                      {pipelineRun.result.metadata.word_count && (
                        <div>
                          <span className="font-medium">Word Count:</span> {pipelineRun.result.metadata.word_count}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        );

      case 'TEXT_CLASSIFIER':
        return (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Brain className="h-5 w-5 mr-2 text-blue-600" />
                  Classification Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                    <div>
                      <h4 className="font-medium text-gray-900">Predicted Category</h4>
                      <p className="text-lg font-semibold text-blue-600">
                        {pipelineRun.result.category || pipelineRun.result.prediction || 'Not available'}
                      </p>
                    </div>
                    {pipelineRun.result.confidence && (
                      <div className="text-right">
                        <h4 className="font-medium text-gray-900">Confidence</h4>
                        <p className="text-lg font-semibold text-green-600">
                          {Math.round(pipelineRun.result.confidence * 100)}%
                        </p>
                      </div>
                    )}
                  </div>

                  {pipelineRun.result.probabilities && (
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-3">All Category Probabilities</h4>
                      <div className="space-y-2">
                        {Object.entries(pipelineRun.result.probabilities).map(([category, probability]) => (
                          <div key={category} className="flex items-center justify-between">
                            <span className="text-sm text-gray-700">{category}</span>
                            <div className="flex items-center space-x-2">
                              <div className="w-24 bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-blue-600 h-2 rounded-full" 
                                  style={{ width: `${(probability as number) * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-sm font-medium text-gray-900">
                                {Math.round((probability as number) * 100)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        );

      default:
        return (
          <Card>
            <CardHeader>
              <CardTitle>Pipeline Results</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-50 p-4 rounded-lg overflow-auto text-sm">
                {JSON.stringify(pipelineRun.result, null, 2)}
              </pre>
            </CardContent>
          </Card>
        );
    }
  };

  const handleExportResults = () => {
    if (!pipelineRun) return;

    const dataStr = JSON.stringify(pipelineRun.result, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `pipeline_results_${pipelineRun.run_uuid.slice(0, 8)}_${pipelineRun.pipeline_type}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  if (isLoading) {
    return (
      <PageLayout>
        <div className="flex justify-center items-center py-12">
          <Spinner size="lg" />
        </div>
      </PageLayout>
    );
  }

  if (error || !pipelineRun) {
    return (
      <PageLayout>
        <div className="text-center py-12">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
          <p className="text-gray-600 mb-6">{error || 'Pipeline not found'}</p>
          <Button onClick={() => navigate(-1)}>Go Back</Button>
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
          onClick={() => navigate(-1)}
        >
          Back
        </Button>

        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              {pipelineRun.pipeline_type.replace('_', ' ')} Results
            </h1>
            <p className="text-gray-600 mt-2">
              Pipeline completed on {formatDate(pipelineRun.updated_at)}
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              onClick={handleExportResults}
              className="text-sm"
            >
              <Download className="h-4 w-4 mr-1" />
              Export Results
            </Button>
          </div>
        </div>

        {/* Status Banner */}
        <div className={`p-4 rounded-lg border ${
          pipelineRun.status === 'COMPLETED' 
            ? 'bg-green-50 border-green-200 text-green-800'
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
          <div className="flex items-center">
            <Eye className="h-5 w-5 mr-2" />
            <span className="font-medium">
              Pipeline {pipelineRun.status.toLowerCase()}
            </span>
            <span className="ml-2">
              • Run ID: {pipelineRun.run_uuid.slice(0, 8)}...
            </span>
          </div>
        </div>

        {/* Results */}
        {pipelineRun.status === 'COMPLETED' ? (
          renderPipelineResults()
        ) : (
          <Card>
            <CardContent className="p-6 text-center">
              <AlertCircle className="h-8 w-8 text-amber-500 mx-auto mb-3" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Pipeline Not Completed</h3>
              <p className="text-gray-600">
                This pipeline has not completed successfully. Status: {pipelineRun.status}
              </p>
              {pipelineRun.error_message && (
                <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                  <strong>Error:</strong> {pipelineRun.error_message}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default PipelineResultsPage; 