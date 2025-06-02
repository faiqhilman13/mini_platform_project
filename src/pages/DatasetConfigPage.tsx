import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { 
  ChevronLeft, 
  Database, 
  Target, 
  Settings,
  ArrowRight, 
  AlertCircle,
  CheckCircle,
  Info,
  Zap,
  BarChart3,
  AlertTriangle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import PageLayout from '../components/layout/PageLayout';
import DatasetPreview from '../components/ml/DatasetPreview';
import DataProfileSummary from '../components/ml/DataProfileSummary';
import Button from '../components/ui/Button';
import Tabs from '../components/ui/Tabs';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '../components/ui/Card';
import {   DatasetProfileSummary as DatasetProfileType,   UploadedFile,  PreprocessingConfig} from '../types';
import { 
  getDatasetPreview, 
  getDatasetProfile,
  getUploadedFiles,
  triggerPipeline,
  triggerMLPipeline
} from '../services/api';

interface FeatureSelectionProps {
  columns: string[];
  selectedFeatures: string[];
  targetColumn: string;
  onFeaturesChange: (features: string[]) => void;
  recommendations?: string[];
  profile?: DatasetProfileType;
}

const FeatureSelection = ({ 
  columns, 
  selectedFeatures, 
  targetColumn,
  onFeaturesChange,
  recommendations = [],
  profile
}: FeatureSelectionProps) => {
  const availableFeatures = columns.filter(col => col !== targetColumn);
  
  // Define high cardinality threshold (matches backend)
  const HIGH_CARDINALITY_THRESHOLD = 20;
  
  // Get high cardinality features from profile
  const getHighCardinalityFeatures = () => {
    if (!profile?.columns) return [];
    return profile.columns
      .filter(col => col.name !== targetColumn && col.unique_count > HIGH_CARDINALITY_THRESHOLD)
      .map(col => col.name);
  };

  const highCardinalityFeatures = getHighCardinalityFeatures();
  const selectedHighCardinalityFeatures = selectedFeatures.filter(feature => 
    highCardinalityFeatures.includes(feature)
  );
  
  const toggleFeature = (feature: string) => {
    if (selectedFeatures.includes(feature)) {
      onFeaturesChange(selectedFeatures.filter(f => f !== feature));
    } else {
      onFeaturesChange([...selectedFeatures, feature]);
    }
  };

  const selectAll = () => {
    onFeaturesChange(availableFeatures);
  };

  const selectNone = () => {
    onFeaturesChange([]);
  };

  const selectRecommended = () => {
    onFeaturesChange(recommendations.filter(rec => availableFeatures.includes(rec)));
  };

  const getFeatureUniqueCount = (featureName: string): number | null => {
    if (!profile?.columns) return null;
    const column = profile.columns.find(col => col.name === featureName);
    return column?.unique_count || null;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
          Feature Selection ({selectedFeatures.length} of {availableFeatures.length} selected)
        </h3>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={selectNone}
            className="text-xs"
          >
            None
          </Button>
          {recommendations.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={selectRecommended}
              className="text-xs"
            >
              <Zap className="h-3 w-3 mr-1" />
              Recommended
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={selectAll}
            className="text-xs"
          >
            All
          </Button>
        </div>
      </div>

      {/* High Cardinality Warning */}
      {selectedHighCardinalityFeatures.length > 0 && (
        <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
          <div className="flex items-start">
            <AlertTriangle className="h-4 w-4 text-orange-600 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-orange-800">High Cardinality Features Selected</p>
              <p className="text-xs text-orange-700 mt-1">
                The following features have many unique values and will be automatically transformed:
              </p>
              <ul className="text-xs text-orange-700 mt-2 space-y-1">
                {selectedHighCardinalityFeatures.map(feature => {
                  const uniqueCount = getFeatureUniqueCount(feature);
                  return (
                    <li key={feature} className="flex items-center">
                      <span className="font-medium">{feature}</span>
                      <span className="ml-2 text-orange-600">
                        ({uniqueCount} unique values → will use label encoding instead of one-hot)
                      </span>
                    </li>
                  );
                })}
              </ul>
              <p className="text-xs text-orange-600 mt-2 font-medium">
                💡 This is normal - your features will still be trained on as you selected.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
        {availableFeatures.map((feature) => {
          const isSelected = selectedFeatures.includes(feature);
          const isRecommended = recommendations.includes(feature);
          const isHighCardinality = highCardinalityFeatures.includes(feature);
          const uniqueCount = getFeatureUniqueCount(feature);
          
          return (
            <motion.div
              key={feature}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <button
                onClick={() => toggleFeature(feature)}
                className={`w-full p-3 text-left border rounded-lg transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 text-blue-900'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                } ${isRecommended ? 'ring-2 ring-yellow-200' : ''} ${
                  isHighCardinality ? 'border-l-4 border-l-orange-400' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium truncate">{feature}</span>
                  <div className="flex items-center space-x-1">
                    {isRecommended && (
                      <Zap className="h-3 w-3 text-yellow-500" />
                    )}
                    {isHighCardinality && (
                      <AlertTriangle className="h-3 w-3 text-orange-500" />
                    )}
                    {isSelected && (
                      <CheckCircle className="h-4 w-4 text-blue-500" />
                    )}
                  </div>
                </div>
                <div className="mt-1 space-y-1">
                  {isRecommended && (
                    <p className="text-xs text-yellow-700">Recommended feature</p>
                  )}
                  {isHighCardinality && (
                    <p className="text-xs text-orange-700">
                      High cardinality ({uniqueCount} unique values)
                    </p>
                  )}
                  {uniqueCount && !isHighCardinality && (
                    <p className="text-xs text-gray-500">
                      {uniqueCount} unique values
                    </p>
                  )}
                </div>
              </button>
            </motion.div>
          );
        })}
      </div>

      {recommendations.length > 0 && (
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-start">
            <Info className="h-4 w-4 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-yellow-800">Feature Recommendations</p>
              <p className="text-xs text-yellow-700 mt-1">
                Based on data analysis, these features are likely to be most informative for your model.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* General High Cardinality Info */}
      {highCardinalityFeatures.length > 0 && selectedHighCardinalityFeatures.length === 0 && (
        <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="flex items-start">
            <Info className="h-4 w-4 text-gray-600 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-gray-800">High Cardinality Features Available</p>
              <p className="text-xs text-gray-700 mt-1">
                Some features have many unique values ({highCardinalityFeatures.join(', ')}). 
                If selected, they'll be transformed using label encoding for optimal performance.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const DatasetConfigPage = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  
  console.log('DatasetConfigPage mounted with fileId:', fileId);
  
  // File and data state
  const [file, setFile] = useState<UploadedFile | null>(null);
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [profile, setProfile] = useState<DatasetProfileType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Configuration state
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [problemType, setProblemType] = useState<'CLASSIFICATION' | 'REGRESSION'>('CLASSIFICATION');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [autoDetectedType, setAutoDetectedType] = useState<'CLASSIFICATION' | 'REGRESSION' | null>(null);
  
  // UI state
  const [activeTab, setActiveTab] = useState('profile');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Load file information
  const loadFileInfo = useCallback(async () => {
    if (!fileId) return;
    
    try {
      const files = await getUploadedFiles();
      const currentFile = files.find(f => f.id === fileId);
      if (!currentFile) {
        setError('File not found');
        return;
      }
      setFile(currentFile);
    } catch (err) {
      setError('Failed to load file information');
      console.error('Error loading file info:', err);
    }
  }, [fileId]);

  // Load dataset preview and profile
  const loadDatasetData = useCallback(async () => {
    if (!fileId) return;
    
    try {
      setLoading(true);
      
      console.log('Loading dataset data for fileId:', fileId);
      
      // Load preview data
      console.log('Fetching preview data...');
      const preview = await getDatasetPreview(fileId, 20);
      console.log('Preview data received:', preview);
      setPreviewData(preview);
      
      // Load profile data
      console.log('Fetching profile data...');
      const profileData = await getDatasetProfile(fileId);
      console.log('Profile data received:', profileData);
      setProfile(profileData);
      
      // Auto-configure based on profile
      if (profileData?.columns && profileData.columns.length > 0) {
        console.log('Auto-configuring with columns:', profileData.columns);
        // Use the last column as target by default, or use URL param
        const urlTargetColumn = searchParams.get('targetColumn');
        const defaultTarget = urlTargetColumn || profileData.columns[profileData.columns.length - 1].name;
        setTargetColumn(defaultTarget);
        
        // Auto-detect problem type based on target column
        const targetCol = profileData.columns.find(col => col.name === defaultTarget);
        if (targetCol) {
          const autoType = detectProblemType(targetCol, profileData);
          setAutoDetectedType(autoType);
          setProblemType(autoType);
        }
        
        // Select all other columns as features by default
        const features = profileData.columns
          .filter(col => col.name !== defaultTarget)
          .map(col => col.name);
        setSelectedFeatures(features);
      } else {
        console.log('No columns found in profile data');
      }
      
      setError(null);
    } catch (err) {
      console.error('Error loading dataset data:', err);
      setError('Failed to load dataset information');
    } finally {
      setLoading(false);
    }
  }, [fileId, searchParams]);

  // Detect problem type based on column characteristics
  const detectProblemType = (column: any, profileData: DatasetProfileType): 'CLASSIFICATION' | 'REGRESSION' => {
    // If column has very few unique values relative to total, likely classification
    const uniqueRatio = column.unique_count / profileData.total_rows;
    
    // If it's clearly numeric and has many unique values, likely regression
    if (['int', 'float', 'number'].includes(column.data_type.toLowerCase())) {
      // If it has more than 20 unique values or more than 10% unique ratio, likely regression
      if (column.unique_count > 20 || uniqueRatio > 0.1) {
        return 'REGRESSION';
      }
    }
    
    // If it has very few unique values, likely classification
    if (column.unique_count <= 10 || uniqueRatio <= 0.05) {
      return 'CLASSIFICATION';
    }
    
    // For exam scores, test scores, grades, etc. - these are typically regression
    const columnNameLower = column.name.toLowerCase();
    if (columnNameLower.includes('score') || columnNameLower.includes('grade') || 
        columnNameLower.includes('points') || columnNameLower.includes('rating')) {
      return 'REGRESSION';
    }
    
    // Default to classification for ambiguous cases
    return 'CLASSIFICATION';
  };

  // Get feature recommendations based on data analysis
  const getFeatureRecommendations = (): string[] => {
    if (!profile?.columns) return [];
    
    return profile.columns
      .filter(col => {
        // Exclude target column
        if (col.name === targetColumn) return false;
        
        // Exclude columns with too many missing values
        const missingRatio = col.missing_count / profile.total_rows;
        if (missingRatio > 0.5) return false;
        
        // Exclude columns with very low variance (likely constant)
        const uniqueRatio = col.unique_count / profile.total_rows;
        if (uniqueRatio < 0.01) return false;
        
        return true;
      })
      .map(col => col.name);
  };

  // Validate configuration
  const validateConfiguration = (): string[] => {
    const errors: string[] = [];
    
    if (!targetColumn) {
      errors.push('Please select a target column');
    }
    
    if (selectedFeatures.length === 0) {
      errors.push('Please select at least one feature column');
    }
    
    if (profile?.columns) {
      const targetCol = profile.columns.find(col => col.name === targetColumn);
      if (targetCol) {
        const missingRatio = targetCol.missing_count / profile.total_rows;
        if (missingRatio > 0.3) {
          errors.push(`Target column "${targetColumn}" has ${(missingRatio * 100).toFixed(1)}% missing values`);
        }
      }
    }
    
    return errors;
  };

  // Get appropriate algorithms for the problem type
  const getAlgorithmsForProblemType = (problemType: 'CLASSIFICATION' | 'REGRESSION') => {
    if (problemType === 'CLASSIFICATION') {
      return [
        'logistic_regression',
        'random_forest_classifier', 
        'decision_tree_classifier'
      ];
    } else {
      return [
        'linear_regression',
        'random_forest_regressor',
        'decision_tree_regressor'
      ];
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    console.log('Starting handleSubmit with:', {
      targetColumn,
      problemType,
      selectedFeatures,
      fileId
    });
    
    const errors = validateConfiguration();
    setValidationErrors(errors);
    
    if (errors.length > 0) {
      console.log('Validation errors:', errors);
      return;
    }
    
    if (!fileId) {
      console.error('No fileId available');
      return;
    }
    
    if (!targetColumn) {
      console.error('No target column selected');
      setError('Please select a target column');
      return;
    }
    
    try {
      setIsSubmitting(true);
      
      // Create basic ML pipeline configuration
      const algorithmNames = getAlgorithmsForProblemType(problemType);
      const algorithms = algorithmNames.map(name => ({
        name,
        hyperparameters: name.includes('random_forest') ? { n_estimators: 100 } : {}
      }));

      const preprocessingConfig = {
        missing_strategy: 'mean',
        categorical_strategy: 'onehot',
        scaling_strategy: 'standard',
        test_size: 0.2,
        feature_columns: selectedFeatures // Include selected features
      };

      const config = {
        target_variable: targetColumn,
        target_column: targetColumn,  // Alternative format
        target: targetColumn,         // Another alternative format
        problem_type: problemType.toLowerCase(),
        algorithms: algorithms,
        preprocessing_config: {
          ...preprocessingConfig,
          target_variable: targetColumn,
          target_column: targetColumn,
          target: targetColumn
        },
        experiment_name: `ML Training - ${file?.filename}`,
        experiment_description: `${problemType} model training with ${selectedFeatures.length} features`,
        feature_columns: selectedFeatures,
        selected_features: selectedFeatures  // Alternative format
      };
      
      console.log('Submitting configuration:', config);

      // Try ML-specific pipeline first, then fallback to general pipeline
      let result;
      try {
        // Try the ML-specific endpoint first
        result = await triggerMLPipeline(fileId, config);
        console.log('ML Pipeline triggered successfully:', result);
      } catch (mlError) {
        console.log('ML pipeline failed, trying general pipeline:', mlError);
        // Fallback to general pipeline
        result = await triggerPipeline(fileId, 'ML_TRAINING', config);
        console.log('General pipeline triggered successfully:', result);
      }
      
      // Add detailed logging for navigation
      console.log('Pipeline result:', result);
      
      if (!result) {
        throw new Error('No result returned from pipeline');
      }
      
      if (!result.run_uuid) {
        console.error('No run_uuid in result:', result);
        throw new Error('Pipeline started but no run ID returned');
      }
      
      const navigationUrl = `/pipeline-results/${result.run_uuid}`;
      console.log('Navigating to:', navigationUrl);
      
      // Navigate to results page
      navigate(navigationUrl);
      
    } catch (err) {
      console.error('Error starting ML training:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(`Failed to start ML training: ${errorMessage}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    loadFileInfo();
    loadDatasetData();
  }, [loadFileInfo, loadDatasetData]);

  if (loading) {
    return (
      <PageLayout>
        <div className="flex items-center justify-center min-h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-500">Loading dataset information...</p>
          </div>
        </div>
      </PageLayout>
    );
  }

  if (error) {
    return (
      <PageLayout>
        <div className="flex items-center justify-center min-h-64">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-600">{error}</p>
            <Button 
              onClick={() => navigate('/upload')} 
              className="mt-4"
              variant="outline"
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              Back to Upload
            </Button>
          </div>
        </div>
      </PageLayout>
    );
  }

  

  return (
    <PageLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate('/upload')}
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Dataset Configuration</h1>
              {file && (
                <p className="text-gray-300">{file.filename}</p>
              )}
            </div>
          </div>
          
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || !targetColumn || selectedFeatures.length === 0}
            className="min-w-32"
          >
            {isSubmitting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Starting...
              </>
            ) : (
              <>
                Start Training
                <ArrowRight className="h-4 w-4 ml-2" />
              </>
            )}
          </Button>
        </div>

        {/* Configuration Summary */}
        {(targetColumn || selectedFeatures.length > 0) && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="bg-blue-50 border-blue-200">
              <CardContent className="pt-4">
                <div className="flex items-center space-x-6 text-sm">
                  <div className="flex items-center">
                    <Target className="h-4 w-4 text-blue-600 mr-2" />
                    <span className="text-gray-700 dark:text-gray-300">Target:</span>
                    <span className="font-medium text-blue-900 dark:text-blue-300 ml-1">
                      {targetColumn || 'Not selected'}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-gray-700 dark:text-gray-300">Type:</span>
                    <span className={`font-medium ml-1 ${
                      problemType === 'CLASSIFICATION' ? 'text-purple-700 dark:text-purple-300' : 'text-green-700 dark:text-green-300'
                    }`}>
                      {problemType}
                      {autoDetectedType && autoDetectedType === problemType && (
                        <span className="text-xs text-gray-500 ml-1">(auto-detected)</span>
                      )}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-gray-700 dark:text-gray-300">Features:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100 ml-1">
                      {selectedFeatures.length} selected
                    </span>
                  </div>
                </div>
                
                {/* Preprocessing Preview */}
                {selectedFeatures.length > 0 && profile && (
                  <div className="mt-4 pt-4 border-t border-blue-200">
                    <h4 className="text-sm font-medium text-blue-900 mb-2">Preprocessing Preview:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div>
                        <span className="font-medium text-blue-800">Numerical features:</span>
                        <div className="text-blue-700 mt-1">
                          {selectedFeatures.filter(feature => {
                            const column = profile.columns?.find(col => col.name === feature);
                            return column && ['int64', 'float64', 'number'].includes(column.data_type.toLowerCase());
                          }).map(feature => (
                            <div key={feature} className="flex items-center justify-between">
                              <span>{feature}</span>
                              <span className="text-blue-600">→ Standard scaling</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <span className="font-medium text-blue-800">Categorical features:</span>
                        <div className="text-blue-700 mt-1">
                          {selectedFeatures.filter(feature => {
                            const column = profile.columns?.find(col => col.name === feature);
                            return column && ['object', 'string', 'category'].includes(column.data_type.toLowerCase());
                          }).map(feature => {
                            const column = profile.columns?.find(col => col.name === feature);
                            const isHighCardinality = column && column.unique_count > 20;
                            return (
                              <div key={feature} className="flex items-center justify-between">
                                <span className={isHighCardinality ? 'font-medium' : ''}>{feature}</span>
                                <span className={`${isHighCardinality ? 'text-orange-600 font-medium' : 'text-blue-600'}`}>
                                  → {isHighCardinality ? 'Label encoding' : 'One-hot encoding'}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Validation Errors */}
        <AnimatePresence>
          {validationErrors.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              <Card className="border-red-200 bg-red-50">
                <CardContent className="pt-4">
                  <div className="flex items-start">
                    <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                    <div>
                      <h3 className="text-sm font-medium text-red-800">Configuration Issues</h3>
                      <ul className="mt-2 text-sm text-red-700 space-y-1">
                        {validationErrors.map((error, index) => (
                          <li key={index}>• {error}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

                {/* Main Content */}        <Tabs           tabs={[            {              id: 'profile',              label: (                <div className="flex items-center">                  <BarChart3 className="h-4 w-4 mr-2" />                  Data Profile                </div>              ),              content: profile ? (                <DataProfileSummary                  profile={profile}                  isLoading={false}                  error={null}                />              ) : null            },            {              id: 'preview',              label: (                <div className="flex items-center">                  <Database className="h-4 w-4 mr-2" />                  Data Preview                </div>              ),              content: (                <DatasetPreview                  data={previewData}                  columns={profile?.columns}                  isLoading={false}                  error={null}                  maxRows={20}                  maxCols={10}                  showStatistics={true}                />              )            },            {              id: 'config',              label: (                <div className="flex items-center">                  <Settings className="h-4 w-4 mr-2" />                  Configuration                </div>              ),              content: profile ? (                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">                  {/* Target Variable Selection */}                  <Card>                    <CardHeader>                      <CardTitle className="flex items-center">                        <Target className="h-5 w-5 mr-2 text-purple-600" />                        Target Variable                      </CardTitle>                    </CardHeader>                    <CardContent>                      <div className="space-y-4">                        <div>                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Select the column you want to predict:
                          </label>
                          <select
                            value={targetColumn}
                            onChange={(e) => {
                              setTargetColumn(e.target.value);
                              const targetCol = profile?.columns?.find(col => col.name === e.target.value);
                              if (targetCol) {
                                const detectedType = detectProblemType(targetCol, profile!);
                                setAutoDetectedType(detectedType);
                                setProblemType(detectedType);
                              }
                            }}
                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                          >
                            <option value="">Select target column...</option>
                            {profile?.columns?.map((column) => (
                              <option key={column.name} value={column.name}>
                                {column.name} ({column.data_type}, {column.unique_count} unique)
                              </option>
                            )) || []}
                          </select>                        </div>                        {targetColumn && (                          <div className="space-y-3">                            <div>                              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                Problem Type:
                              </label>
                              <div className="flex space-x-4">                                <label className="flex items-center">                                  <input                                    type="radio"                                    value="CLASSIFICATION"                                    checked={problemType === 'CLASSIFICATION'}                                    onChange={(e) => setProblemType(e.target.value as 'CLASSIFICATION')}                                    className="mr-2"                                  />                                  Classification                                </label>                                <label className="flex items-center">                                  <input                                    type="radio"                                    value="REGRESSION"                                    checked={problemType === 'REGRESSION'}                                    onChange={(e) => setProblemType(e.target.value as 'REGRESSION')}                                    className="mr-2"                                  />                                  Regression                                </label>                              </div>                            </div>                            {autoDetectedType && (                              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">                                <div className="flex items-start">                                  <Info className="h-4 w-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />                                  <div>                                    <p className="text-sm font-medium text-blue-800">                                      Auto-detected: {autoDetectedType}                                    </p>                                    <p className="text-xs text-blue-700 mt-1">                                      Based on the target column characteristics. You can change this if needed.                                    </p>                                  </div>                                </div>                              </div>                            )}                          </div>                        )}                      </div>                    </CardContent>                  </Card>                  {/* Feature Selection */}                  <Card>                    <CardHeader>                      <CardTitle className="flex items-center">                        <Settings className="h-5 w-5 mr-2 text-green-600" />                        Feature Selection                      </CardTitle>                    </CardHeader>                    <CardContent>                      {targetColumn ? (                        <FeatureSelection                          columns={profile?.columns?.map(col => col.name) || []}                          selectedFeatures={selectedFeatures}                          targetColumn={targetColumn}                          onFeaturesChange={setSelectedFeatures}                          recommendations={getFeatureRecommendations()}                          profile={profile}                        />                      ) : (                        <div className="text-center py-8 text-gray-500">                          Please select a target column first                        </div>                      )}                    </CardContent>                  </Card>                </div>              ) : null            }          ]}          defaultTabId={activeTab}          onChange={setActiveTab}        />
      </div>
    </PageLayout>
  );
};

export default DatasetConfigPage; 
