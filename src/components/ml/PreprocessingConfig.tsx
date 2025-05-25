import React, { useState, useEffect } from 'react';
import { 
  Cog,
  Database,
  BarChart3,
  Shuffle,
  AlertTriangle,
  Info,
  CheckCircle,
  TrendingUp,
  Layers,
  Filter
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { PreprocessingConfig as PreprocessingConfigType } from '../../types';

interface PreprocessingConfigProps {
  config: PreprocessingConfigType;
  onConfigChange: (config: PreprocessingConfigType) => void;
  datasetInfo?: {
    totalRows: number;
    totalColumns: number;
    missingCells: number;
    categoricalColumns: number;
    numericColumns: number;
  };
  className?: string;
}

interface ConfigSectionProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

const ConfigSection = ({ title, description, icon, children, className }: ConfigSectionProps) => (
  <div className={cn('border border-gray-200 rounded-lg p-4', className)}>
    <div className="flex items-start space-x-3 mb-4">
      <div className="flex-shrink-0 mt-1">
        {icon}
      </div>
      <div className="flex-1">
        <h3 className="text-sm font-medium text-gray-900">{title}</h3>
        <p className="text-xs text-gray-600 mt-1">{description}</p>
      </div>
    </div>
    {children}
  </div>
);

const PreprocessingConfig = ({
  config,
  onConfigChange,
  datasetInfo,
  className
}: PreprocessingConfigProps) => {
  const [validationMessages, setValidationMessages] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Update a specific config field
  const updateConfig = (field: keyof PreprocessingConfigType, value: any) => {
    onConfigChange({
      ...config,
      [field]: value
    });
  };

  // Validate configuration and generate messages
  const validateConfiguration = () => {
    const messages: string[] = [];
    
    // Train/test split validation
    if (config.trainTestSplit < 0.1 || config.trainTestSplit > 0.9) {
      messages.push('Train/test split should be between 10% and 90%');
    }
    
    // Missing value strategy validation
    if (datasetInfo && datasetInfo.missingCells > 0) {
      if (config.missingValueStrategy === 'drop') {
        const estimatedRowsLost = Math.round(datasetInfo.totalRows * (datasetInfo.missingCells / (datasetInfo.totalRows * datasetInfo.totalColumns)));
        if (estimatedRowsLost > datasetInfo.totalRows * 0.3) {
          messages.push(`Dropping rows with missing values may remove ~${estimatedRowsLost} rows (${(estimatedRowsLost/datasetInfo.totalRows*100).toFixed(1)}%)`);
        }
      }
    }
    
    // Feature selection validation
    if (config.featureSelection && config.featureSelection.length === 0) {
      messages.push('No features selected - at least one feature is required');
    }
    
    setValidationMessages(messages);
  };

  // Validate on config changes
  useEffect(() => {
    validateConfiguration();
  }, [config, datasetInfo]);

  // Get recommendations based on data characteristics
  const getRecommendations = () => {
    const recommendations: { type: 'info' | 'warning' | 'success'; message: string }[] = [];
    
    if (datasetInfo) {
      // Missing value recommendations
      if (datasetInfo.missingCells > 0) {
        const missingPercentage = (datasetInfo.missingCells / (datasetInfo.totalRows * datasetInfo.totalColumns)) * 100;
        if (missingPercentage > 20) {
          recommendations.push({
            type: 'warning',
            message: `High missing data (${missingPercentage.toFixed(1)}%). Consider 'mean' or 'median' imputation.`
          });
        } else {
          recommendations.push({
            type: 'info',
            message: `Low missing data (${missingPercentage.toFixed(1)}%). Any imputation strategy should work well.`
          });
        }
      }
      
      // Scaling recommendations
      if (datasetInfo.numericColumns > 0) {
        recommendations.push({
          type: 'info',
          message: 'Standard scaling is recommended for most algorithms to normalize feature ranges.'
        });
      }
      
      // Train/test split recommendations
      if (datasetInfo.totalRows < 1000) {
        recommendations.push({
          type: 'warning',
          message: 'Small dataset detected. Consider using cross-validation or a smaller test split.'
        });
      }
    }
    
    return recommendations;
  };

  const recommendations = getRecommendations();

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Cog className="h-5 w-5 mr-2 text-teal-600" />
            Preprocessing Configuration
          </CardTitle>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs"
          >
            {showAdvanced ? 'Basic' : 'Advanced'}
          </Button>
        </div>
        
        {datasetInfo && (
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <span>{datasetInfo.totalRows} rows</span>
            <span>{datasetInfo.totalColumns} columns</span>
            {datasetInfo.missingCells > 0 && (
              <span className="text-amber-600">
                {datasetInfo.missingCells} missing values
              </span>
            )}
          </div>
        )}
      </CardHeader>
      
      <CardContent>
        <div className="space-y-6">
          {/* Missing Value Handling */}
          <ConfigSection
            title="Missing Value Handling"
            description="How to handle missing values in your dataset"
            icon={<Database className="h-4 w-4 text-blue-600" />}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {[
                { value: 'mean', label: 'Mean', description: 'Replace with column average (numeric only)' },
                { value: 'median', label: 'Median', description: 'Replace with column median (numeric only)' },
                { value: 'mode', label: 'Mode', description: 'Replace with most frequent value' },
                { value: 'constant', label: 'Constant', description: 'Replace with a fixed value' },
                { value: 'drop', label: 'Drop Rows', description: 'Remove rows with missing values' }
              ].map((strategy) => (
                <motion.button
                  key={strategy.value}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => updateConfig('missingValueStrategy', strategy.value as any)}
                  className={cn(
                    'p-3 text-left border rounded-lg transition-all',
                    config.missingValueStrategy === strategy.value
                      ? 'border-blue-500 bg-blue-50 text-blue-900'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  )}
                >
                  <div className="font-medium text-sm">{strategy.label}</div>
                  <div className="text-xs text-gray-600 mt-1">{strategy.description}</div>
                </motion.button>
              ))}
            </div>
            
            {config.missingValueStrategy === 'constant' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3"
              >
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Constant Value
                </label>
                <input
                  type="text"
                  value={config.constantValue || ''}
                  onChange={(e) => updateConfig('constantValue', e.target.value)}
                  placeholder="Enter replacement value"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />
              </motion.div>
            )}
          </ConfigSection>

          {/* Feature Scaling */}
          <ConfigSection
            title="Feature Scaling"
            description="Scale numeric features to improve algorithm performance"
            icon={<BarChart3 className="h-4 w-4 text-green-600" />}
          >
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {[
                { value: 'none', label: 'No Scaling', description: 'Keep original values' },
                { value: 'standard', label: 'Standard Scaling', description: 'Mean = 0, Std = 1' },
                { value: 'minmax', label: 'Min-Max Scaling', description: 'Scale to [0, 1] range' }
              ].map((method) => (
                <motion.button
                  key={method.value}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => updateConfig('scaling', method.value as any)}
                  className={cn(
                    'p-3 text-left border rounded-lg transition-all',
                    config.scaling === method.value
                      ? 'border-green-500 bg-green-50 text-green-900'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  )}
                >
                  <div className="font-medium text-sm">{method.label}</div>
                  <div className="text-xs text-gray-600 mt-1">{method.description}</div>
                </motion.button>
              ))}
            </div>
          </ConfigSection>

          {/* Categorical Encoding */}
          <ConfigSection
            title="Categorical Encoding"
            description="Convert categorical variables to numeric format"
            icon={<Layers className="h-4 w-4 text-purple-600" />}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { value: 'onehot', label: 'One-Hot Encoding', description: 'Create binary columns for each category' },
                { value: 'label', label: 'Label Encoding', description: 'Assign numeric labels to categories' }
              ].map((method) => (
                <motion.button
                  key={method.value}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => updateConfig('categoricalEncoding', method.value as any)}
                  className={cn(
                    'p-3 text-left border rounded-lg transition-all',
                    config.categoricalEncoding === method.value
                      ? 'border-purple-500 bg-purple-50 text-purple-900'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  )}
                >
                  <div className="font-medium text-sm">{method.label}</div>
                  <div className="text-xs text-gray-600 mt-1">{method.description}</div>
                </motion.button>
              ))}
            </div>
          </ConfigSection>

          {/* Train/Test Split */}
          <ConfigSection
            title="Train/Test Split"
            description="Divide data for training and evaluation"
            icon={<Shuffle className="h-4 w-4 text-orange-600" />}
          >
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Test Set Size: {(config.trainTestSplit * 100).toFixed(0)}%
                  </label>
                  <span className="text-xs text-gray-500">
                    Training: {(100 - config.trainTestSplit * 100).toFixed(0)}%
                  </span>
                </div>
                
                <input
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={config.trainTestSplit}
                  onChange={(e) => updateConfig('trainTestSplit', parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>10%</span>
                  <span>50%</span>
                </div>
              </div>
              
              {datasetInfo && (
                <div className="bg-gray-50 p-3 rounded-lg">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Training samples:</span>
                      <span className="font-medium ml-2">
                        {Math.round(datasetInfo.totalRows * (1 - config.trainTestSplit))}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Test samples:</span>
                      <span className="font-medium ml-2">
                        {Math.round(datasetInfo.totalRows * config.trainTestSplit)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ConfigSection>

          {/* Advanced Options */}
          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <ConfigSection
                  title="Advanced Options"
                  description="Additional preprocessing configurations"
                  icon={<Filter className="h-4 w-4 text-indigo-600" />}
                >
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium text-gray-700">
                          Random Seed
                        </label>
                        <p className="text-xs text-gray-600">
                          Set seed for reproducible results
                        </p>
                      </div>
                      <input
                        type="number"
                        value={42}
                        className="w-24 px-3 py-2 border border-gray-300 rounded-md text-sm"
                        readOnly
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium text-gray-700">
                          Stratify Split
                        </label>
                        <p className="text-xs text-gray-600">
                          Maintain class distribution in splits
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={config.problemType === 'CLASSIFICATION'}
                        disabled
                        className="rounded"
                      />
                    </div>
                  </div>
                </ConfigSection>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Recommendations */}
          {recommendations.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-gray-900 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2 text-gray-600" />
                Recommendations
              </h4>
              
              {recommendations.map((rec, index) => (
                <div
                  key={index}
                  className={cn(
                    'p-3 rounded-lg border',
                    rec.type === 'warning' 
                      ? 'bg-amber-50 border-amber-200 text-amber-800'
                      : rec.type === 'success'
                        ? 'bg-green-50 border-green-200 text-green-800'
                        : 'bg-blue-50 border-blue-200 text-blue-800'
                  )}
                >
                  <div className="flex items-start">
                    {rec.type === 'warning' ? (
                      <AlertTriangle className="h-4 w-4 mt-0.5 mr-2 flex-shrink-0" />
                    ) : rec.type === 'success' ? (
                      <CheckCircle className="h-4 w-4 mt-0.5 mr-2 flex-shrink-0" />
                    ) : (
                      <Info className="h-4 w-4 mt-0.5 mr-2 flex-shrink-0" />
                    )}
                    <p className="text-sm">{rec.message}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Validation Messages */}
          {validationMessages.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-red-800 flex items-center">
                <AlertTriangle className="h-4 w-4 mr-2" />
                Configuration Issues
              </h4>
              
              {validationMessages.map((message, index) => (
                <div key={index} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-700">{message}</p>
                </div>
              ))}
            </div>
          )}

          {/* Configuration Summary */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Configuration Summary</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-gray-600">Missing values:</span>
                <span className="font-medium ml-2 capitalize">
                  {config.missingValueStrategy}
                  {config.missingValueStrategy === 'constant' && config.constantValue && 
                    ` (${config.constantValue})`
                  }
                </span>
              </div>
              <div>
                <span className="text-gray-600">Scaling:</span>
                <span className="font-medium ml-2 capitalize">
                  {config.scaling === 'none' ? 'None' : config.scaling}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Categorical encoding:</span>
                <span className="font-medium ml-2 capitalize">
                  {config.categoricalEncoding === 'onehot' ? 'One-hot' : 'Label'}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Test split:</span>
                <span className="font-medium ml-2">
                  {(config.trainTestSplit * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PreprocessingConfig; 