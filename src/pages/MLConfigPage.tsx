import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, Table2, Database, Cog, ArrowRight } from 'lucide-react';
import PageLayout from '../components/layout/PageLayout';
import DatasetPreview from '../components/ml/DatasetPreview';
import DataProfileSummary from '../components/ml/DataProfileSummary';
import PreprocessingForm from '../components/ml/PreprocessingForm';
import AlgorithmSelector from '../components/ml/AlgorithmSelector';
import HyperparameterConfig from '../components/ml/HyperparameterConfig';
import Button from '../components/ui/Button';
import Tabs from '../components/ui/Tabs';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '../components/ui/Card';
import { 
  MLPipelineRun, 
  DatasetProfileSummary, 
  PreprocessingConfig,
  AlgorithmOption 
} from '../types';
import { 
  getPipelineStatus, 
  getDatasetPreview, 
  getDatasetProfile,
  triggerMLPipeline 
} from '../services/api';
import { ML_ALGORITHMS } from '../utils/constants';

const MLConfigPage = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  
  // Pipeline state
  const [pipelineRun, setPipelineRun] = useState<MLPipelineRun | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Dataset state
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  
  const [profile, setProfile] = useState<DatasetProfileSummary | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  
  // Configuration state
  const [preprocessing, setPreprocessing] = useState<PreprocessingConfig>({
    targetColumn: '',
    problemType: 'CLASSIFICATION',
    trainTestSplit: 80,
    missingValueStrategy: 'mean',
    scaling: 'standard',
    categoricalEncoding: 'onehot',
    featureSelection: [],
  });
  
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([]);
  const [hyperParameters, setHyperParameters] = useState<Record<string, Record<string, any>>>({});
  
  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Fetch pipeline run details
  const fetchPipelineRun = useCallback(async () => {
    if (!runId) return;
    
    try {
      setIsLoading(true);
      const run = await getPipelineStatus(runId);
      setPipelineRun(run as MLPipelineRun);
      
      // Ensure this is a ML_TRAINING pipeline
      if (run.pipeline_type !== 'ML_TRAINING') {
        setError('This is not a Machine Learning pipeline');
      } else {
        setError(null);
        
        // Load dataset preview and profile
        if (run.uploaded_file_log_id) {
          loadDatasetPreview(run.uploaded_file_log_id);
          loadDatasetProfile(run.uploaded_file_log_id);
        }
      }
    } catch (err) {
      setError('Failed to load pipeline details');
      console.error('Error fetching pipeline run:', err);
    } finally {
      setIsLoading(false);
    }
  }, [runId]);

  // Load dataset preview
  const loadDatasetPreview = async (fileId: string) => {
    try {
      setPreviewLoading(true);
      const data = await getDatasetPreview(fileId, 20);
      setPreviewData(data);
      setPreviewError(null);
    } catch (err) {
      setPreviewError('Failed to load dataset preview');
      console.error('Error loading dataset preview:', err);
    } finally {
      setPreviewLoading(false);
    }
  };

  // Load dataset profile
  const loadDatasetProfile = async (fileId: string) => {
    try {
      setProfileLoading(true);
      const profileData = await getDatasetProfile(fileId);
      setProfile(profileData);
      
      // Initialize preprocessing config with profile data
      if (profileData.columns.length > 0) {
        // Use the last column as the target by default
        const lastColumn = profileData.columns[profileData.columns.length - 1];
        
        // Determine problem type based on data type of target column
        const problemType = ['int', 'float', 'number'].includes(lastColumn.data_type) 
          ? 'REGRESSION' 
          : 'CLASSIFICATION';
        
        // Set all columns except target as features
        const features = profileData.columns
          .filter(col => col.name !== lastColumn.name)
          .map(col => col.name);
        
        setPreprocessing({
          ...preprocessing,
          targetColumn: lastColumn.name,
          problemType,
          featureSelection: features,
        });
      }
      
      setProfileError(null);
    } catch (err) {
      setProfileError('Failed to load dataset profile');
      console.error('Error loading dataset profile:', err);
    } finally {
      setProfileLoading(false);
    }
  };

  // Initialize config when algorithms are selected
  useEffect(() => {
    // Initialize hyperparameters with default values for selected algorithms
    const newHyperParams: Record<string, Record<string, any>> = {};
    
    selectedAlgorithms.forEach(algoName => {
      const algorithm = ML_ALGORITHMS.find(a => a.name === algoName);
      if (algorithm) {
        newHyperParams[algoName] = algorithm.hyperparameters.reduce((acc, param) => {
          acc[param.name] = param.default;
          return acc;
        }, {} as Record<string, any>);
      }
    });
    
    setHyperParameters(newHyperParams);
  }, [selectedAlgorithms]);

  useEffect(() => {
    fetchPipelineRun();
  }, [fetchPipelineRun]);

  // Handle submitting the configuration
  const handleSubmit = async () => {
    if (!runId || !pipelineRun?.uploaded_file_log_id) return;
    
    if (selectedAlgorithms.length === 0) {
      setSubmitError('Please select at least one algorithm');
      return;
    }
    
    if (!preprocessing.targetColumn) {
      setSubmitError('Please select a target column');
      return;
    }
    
    try {
      setIsSubmitting(true);
      setSubmitError(null);
      
      // Prepare algorithms config
      const algorithmsConfig: Record<string, any> = {};
      selectedAlgorithms.forEach(algoName => {
        algorithmsConfig[algoName] = hyperParameters[algoName] || {};
      });
      
      // Prepare ML pipeline config
      const mlConfig = {
        preprocessing: preprocessing,
        algorithms: algorithmsConfig,
        problem_type: preprocessing.problemType
      };
      
      // Trigger ML pipeline with configuration
      const mlPipelineRun = await triggerMLPipeline(
        pipelineRun.uploaded_file_log_id,
        mlConfig
      );
      
      // Navigate to results page
      navigate(`/ml/${mlPipelineRun.run_uuid}/results`);
    } catch (err) {
      setSubmitError('Failed to submit ML pipeline configuration');
      console.error('Error submitting ML pipeline config:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Render algorithm configuration
  const renderAlgorithmConfig = () => {
    if (selectedAlgorithms.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          Please select at least one algorithm to configure.
        </div>
      );
    }
    
    return (
      <div className="space-y-6">
        {selectedAlgorithms.map(algoName => {
          const algorithm = ML_ALGORITHMS.find(a => a.name === algoName);
          if (!algorithm) return null;
          
          return (
            <HyperparameterConfig
              key={algoName}
              algorithmName={algoName}
              displayName={algorithm.displayName}
              hyperparameters={algorithm.hyperparameters}
              values={hyperParameters[algoName] || {}}
              onChange={(values) => {
                setHyperParameters(prev => ({
                  ...prev,
                  [algoName]: values
                }));
              }}
            />
          );
        })}
      </div>
    );
  };

  return (
    <PageLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            icon={<ChevronLeft className="w-4 h-4" />}
            onClick={() => navigate(-1)}
          >
            Back
          </Button>
          
          <h1 className="text-2xl font-bold text-gray-900">
            Configure ML Pipeline
          </h1>
          
          <div className="w-24"></div> {/* Spacer for alignment */}
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
          </div>
        ) : error ? (
          <div className="bg-red-50 p-6 rounded-lg text-center">
            <h3 className="text-lg font-medium text-red-800 mb-2">{error}</h3>
            <Button
              variant="outline"
              onClick={() => navigate(-1)}
            >
              Go Back
            </Button>
          </div>
        ) : (
          <div className="space-y-8">
            <DatasetPreview 
              data={previewData}
              isLoading={previewLoading}
              error={previewError}
            />
            
            <DataProfileSummary
              profile={profile}
              isLoading={profileLoading}
              error={profileError}
            />
            
            <Tabs
              tabs={[
                {
                  id: 'preprocessing',
                  label: (
                    <div className="flex items-center">
                      <Database className="w-4 h-4 mr-2" />
                      Data Preprocessing
                    </div>
                  ),
                  content: (
                    <PreprocessingForm
                      profile={profile}
                      config={preprocessing}
                      onChange={setPreprocessing}
                    />
                  )
                },
                {
                  id: 'algorithms',
                  label: (
                    <div className="flex items-center">
                      <Table2 className="w-4 h-4 mr-2" />
                      Algorithm Selection
                    </div>
                  ),
                  content: (
                    <AlgorithmSelector
                      algorithms={ML_ALGORITHMS}
                      problemType={preprocessing.problemType}
                      selectedAlgorithms={selectedAlgorithms}
                      onSelect={setSelectedAlgorithms}
                    />
                  )
                },
                {
                  id: 'hyperparameters',
                  label: (
                    <div className="flex items-center">
                      <Cog className="w-4 h-4 mr-2" />
                      Hyperparameters
                    </div>
                  ),
                  content: renderAlgorithmConfig()
                }
              ]}
            />
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-1">
                      Configuration Summary
                    </h3>
                    <p className="text-sm text-gray-500">
                      Review your configuration before starting the ML pipeline
                    </p>
                  </div>
                  
                  <Button
                    onClick={handleSubmit}
                    disabled={isSubmitting || selectedAlgorithms.length === 0}
                    isLoading={isSubmitting}
                    icon={<ArrowRight className="w-4 h-4" />}
                  >
                    Train Models
                  </Button>
                </div>
                
                {submitError && (
                  <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
                    {submitError}
                  </div>
                )}
                
                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Data</h4>
                    <ul className="text-sm space-y-1">
                      <li><span className="font-medium">Target:</span> {preprocessing.targetColumn}</li>
                      <li><span className="font-medium">Problem:</span> {preprocessing.problemType}</li>
                      <li><span className="font-medium">Train Split:</span> {preprocessing.trainTestSplit}%</li>
                      <li><span className="font-medium">Features:</span> {preprocessing.featureSelection.length}</li>
                    </ul>
                  </div>
                  
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Preprocessing</h4>
                    <ul className="text-sm space-y-1">
                      <li><span className="font-medium">Missing Values:</span> {preprocessing.missingValueStrategy}</li>
                      <li><span className="font-medium">Scaling:</span> {preprocessing.scaling}</li>
                      <li><span className="font-medium">Categorical Encoding:</span> {preprocessing.categoricalEncoding}</li>
                    </ul>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Algorithms</h4>
                    <ul className="text-sm space-y-1">
                      {selectedAlgorithms.map(algo => {
                        const algorithm = ML_ALGORITHMS.find(a => a.name === algo);
                        return (
                          <li key={algo}>{algorithm?.displayName || algo}</li>
                        );
                      })}
                      {selectedAlgorithms.length === 0 && (
                        <li className="text-amber-600">No algorithms selected</li>
                      )}
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default MLConfigPage;
