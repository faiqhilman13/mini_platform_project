import React, { useState } from 'react';
import { 
  Brain,
  BarChart3,
  Target,
  Clock,
  Settings,
  Info,
  Download,
  Eye,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Hash,
  Layers,
  Activity
} from 'lucide-react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { MLModel } from '../../types';

interface ModelDetailsProps {
  model: MLModel;
  problemType: 'CLASSIFICATION' | 'REGRESSION';
  isBest?: boolean;
  className?: string;
}

interface MetricDisplayProps {
  label: string;
  value: string | number;
  description?: string;
  trend?: 'positive' | 'negative' | 'neutral';
  format?: 'percentage' | 'decimal' | 'time' | 'count';
}

const MetricDisplay = ({ label, value, description, trend, format }: MetricDisplayProps) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'percentage':
        return `${(val * 100).toFixed(2)}%`;
      case 'decimal':
        return val.toFixed(4);
      case 'time':
        return `${val.toFixed(2)}s`;
      case 'count':
        return Math.round(val).toString();
            default:        return typeof val === 'number' ? val.toFixed(4) : String(val);
    }
  };

  const getTrendColor = () => {
    switch (trend) {
      case 'positive':
        return 'text-green-600';
      case 'negative':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="bg-gray-50 p-3 rounded-lg">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        {description && (
          <div className="group relative">
            <Info className="h-3 w-3 text-gray-400 cursor-help" />
            <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block z-10">
              <div className="bg-gray-900 text-white text-xs rounded py-1 px-2 max-w-xs whitespace-nowrap">
                {description}
              </div>
            </div>
          </div>
        )}
      </div>
      <div className={cn('text-lg font-bold', getTrendColor())}>
        {formatValue(value)}
      </div>
    </div>
  );
};

interface FeatureImportanceProps {
  featureImportance: Record<string, number>;
  maxFeatures?: number;
}

const FeatureImportance = ({ featureImportance, maxFeatures = 10 }: FeatureImportanceProps) => {
  const sortedFeatures = Object.entries(featureImportance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, maxFeatures);

  const maxImportance = Math.max(...sortedFeatures.map(([, importance]) => importance));

  return (
    <div className="space-y-3">
      {sortedFeatures.map(([feature, importance], index) => (
        <motion.div
          key={feature}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="flex items-center space-x-3"
        >
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium text-gray-700 truncate">
                {feature}
              </span>
              <span className="text-xs text-gray-500">
                {(importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(importance / maxImportance) * 100}%` }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full"
              />
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

interface ConfusionMatrixProps {
  confusionMatrix: number[][];
  classLabels?: string[];
}

const ConfusionMatrix = ({ confusionMatrix, classLabels }: ConfusionMatrixProps) => {
  if (!confusionMatrix || confusionMatrix.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <AlertCircle className="h-8 w-8 text-gray-300 mx-auto mb-2" />
        <p>Confusion matrix not available</p>
      </div>
    );
  }

  const maxValue = Math.max(...confusionMatrix.flat());
  const labels = classLabels || confusionMatrix.map((_, i) => `Class ${i}`);

  return (
    <div className="space-y-4">
      <div className="text-center">
        <div className="inline-block">
          <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${confusionMatrix.length + 1}, minmax(0, 1fr))` }}>
            {/* Header row */}
            <div></div>
            {labels.map((label, i) => (
              <div key={i} className="text-xs font-medium text-gray-600 p-2 text-center">
                {label}
              </div>
            ))}
            
            {/* Matrix rows */}
            {confusionMatrix.map((row, i) => (
              <React.Fragment key={i}>
                <div className="text-xs font-medium text-gray-600 p-2 flex items-center">
                  {labels[i]}
                </div>
                {row.map((value, j) => (
                  <div
                    key={j}
                    className={cn(
                      'p-2 text-center text-sm font-medium rounded',
                      value === 0 ? 'bg-gray-100 text-gray-400' : 'text-white'
                    )}
                    style={{
                      backgroundColor: value === 0 ? undefined : 
                        `rgba(79, 70, 229, ${0.3 + (value / maxValue) * 0.7})`
                    }}
                  >
                    {value}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
      
      <div className="text-center">
        <p className="text-xs text-gray-500">
          Rows: Actual • Columns: Predicted
        </p>
      </div>
    </div>
  );
};

interface HyperparameterDisplayProps {
  hyperparameters: Record<string, any>;
}

const HyperparameterDisplay = ({ hyperparameters }: HyperparameterDisplayProps) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {Object.entries(hyperparameters).map(([key, value]) => (
        <div key={key} className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs font-medium text-gray-600 uppercase tracking-wide mb-1">
            {key.replace(/_/g, ' ')}
          </div>
          <div className="text-sm font-mono text-gray-900">
            {typeof value === 'boolean' ? (value ? 'True' : 'False') : 
             typeof value === 'object' ? JSON.stringify(value) : 
             value.toString()}
          </div>
        </div>
      ))}
    </div>
  );
};

const ModelDetails = ({ model, problemType, isBest = false, className }: ModelDetailsProps) => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'features' | 'config' | 'insights'>('metrics');

  const metrics = model.performance_metrics;
  const algorithmName = model.algorithm_name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

  // Get metrics based on problem type
  const getMetricsDisplay = () => {
    if (problemType === 'CLASSIFICATION') {
      return [
                 {            label: 'Accuracy',            value: metrics.accuracy || 0,            description: 'Proportion of correct predictions',           trend: (metrics.accuracy || 0) > 0.8 ? 'positive' as const : 'neutral' as const,           format: 'percentage' as const         },
        { 
          label: 'Precision', 
          value: metrics.precision || 0, 
          description: 'True positives / (True positives + False positives)',
          format: 'decimal' as const
        },
        { 
          label: 'Recall', 
          value: metrics.recall || 0, 
          description: 'True positives / (True positives + False negatives)',
          format: 'decimal' as const
        },
        { 
          label: 'F1-Score', 
          value: metrics.f1_score || 0, 
          description: 'Harmonic mean of precision and recall',
          format: 'decimal' as const
        },
        { 
          label: 'ROC AUC', 
          value: metrics.roc_auc || 0, 
          description: 'Area under the ROC curve',
          format: 'decimal' as const
        }
      ];
    } else {
      return [
                 {            label: 'R² Score',            value: metrics.r2 || 0,            description: 'Coefficient of determination',           trend: (metrics.r2 || 0) > 0.8 ? 'positive' as const : 'neutral' as const,           format: 'decimal' as const         },
        { 
          label: 'Mean Absolute Error', 
          value: metrics.mae || 0, 
          description: 'Average absolute difference between predicted and actual',
          trend: 'negative' as const,
          format: 'decimal' as const
        },
        { 
          label: 'Root Mean Square Error', 
          value: metrics.rmse || 0, 
          description: 'Square root of average squared errors',
          trend: 'negative' as const,
          format: 'decimal' as const
        },
        { 
          label: 'Mean Squared Error', 
          value: metrics.mse || 0, 
          description: 'Average of squared errors',
          trend: 'negative' as const,
          format: 'decimal' as const
        }
      ];
    }
  };

  const metricsDisplay = getMetricsDisplay();

  const handleExport = () => {
    const exportData = {
      model_info: {
        algorithm: model.algorithm_name,
        model_id: model.model_id,
        training_time: model.training_time
      },
      hyperparameters: model.hyperparameters,
      performance_metrics: model.performance_metrics,
      feature_importance: model.feature_importance
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `model_${model.algorithm_name}_${model.model_id.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={cn(
              'p-2 rounded-full',
              isBest ? 'bg-yellow-100' : 'bg-purple-100'
            )}>
              <Brain className={cn(
                'h-5 w-5',
                isBest ? 'text-yellow-600' : 'text-purple-600'
              )} />
            </div>
            <div>
              <CardTitle className="flex items-center">
                {algorithmName}
                {isBest && (
                  <span className="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Best Model
                  </span>
                )}
              </CardTitle>
              <p className="text-sm text-gray-600 mt-1">
                Trained in {model.training_time ? model.training_time.toFixed(2) : 'N/A'}s • 
                Model ID: {model.model_id.slice(0, 8)}...
              </p>
            </div>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            className="text-xs"
          >
            <Download className="h-3 w-3 mr-1" />
            Export
          </Button>
        </div>
        
        {/* Tabs */}
        <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
          {[
            { id: 'metrics', label: 'Metrics', icon: <BarChart3 className="h-3 w-3" /> },
            { id: 'features', label: 'Features', icon: <TrendingUp className="h-3 w-3" /> },
            { id: 'config', label: 'Config', icon: <Settings className="h-3 w-3" /> },
            { id: 'insights', label: 'Insights', icon: <Eye className="h-3 w-3" /> }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                'flex items-center space-x-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
                activeTab === tab.id
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              )}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent>
        {activeTab === 'metrics' && (
          <div className="space-y-6">
            {/* Performance Metrics */}
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                <Target className="h-4 w-4 mr-2 text-gray-600" />
                Performance Metrics
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {metricsDisplay.map((metric) => (
                  <MetricDisplay
                    key={metric.label}
                    label={metric.label}
                    value={metric.value}
                    description={metric.description}
                    trend={metric.trend}
                    format={metric.format}
                  />
                ))}
              </div>
            </div>

            {/* Training Information */}
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                <Clock className="h-4 w-4 mr-2 text-gray-600" />
                Training Information
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MetricDisplay
                  label="Training Time"
                  value={model.training_time || 0}
                  format="time"
                />
                <MetricDisplay
                  label="Model Size"
                  value={model.model_path ? '~1.2MB' : 'N/A'} // Placeholder
                />
                <MetricDisplay
                  label="Parameters"
                  value={Object.keys(model.hyperparameters).length}
                  format="count"
                />
              </div>
            </div>

            {/* Confusion Matrix for Classification */}
            {problemType === 'CLASSIFICATION' && metrics.confusion_matrix && (
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                  <Hash className="h-4 w-4 mr-2 text-gray-600" />
                  Confusion Matrix
                </h3>
                <ConfusionMatrix
                  confusionMatrix={metrics.confusion_matrix}
                  classLabels={metrics.class_labels}
                />
              </div>
            )}
          </div>
        )}

        {activeTab === 'features' && (
          <div className="space-y-6">
            {model.feature_importance ? (
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                  <TrendingUp className="h-4 w-4 mr-2 text-gray-600" />
                  Feature Importance
                </h3>
                <FeatureImportance featureImportance={model.feature_importance} />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <TrendingUp className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                <p>Feature importance not available for this algorithm</p>
                <p className="text-xs mt-1">
                  This algorithm doesn't provide feature importance scores
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'config' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                <Settings className="h-4 w-4 mr-2 text-gray-600" />
                Hyperparameters
              </h3>
              <HyperparameterDisplay hyperparameters={model.hyperparameters} />
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                <Layers className="h-4 w-4 mr-2 text-gray-600" />
                Model Information
              </h3>
              <div className="bg-gray-50 p-4 rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Algorithm</span>
                  <span className="text-sm font-medium text-gray-900">{algorithmName}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Model ID</span>
                  <span className="text-sm font-mono text-gray-900">{model.model_id}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Pipeline Run</span>
                  <span className="text-sm font-mono text-gray-900">{model.pipeline_run_id}</span>
                </div>
                {model.model_path && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Model Path</span>
                    <span className="text-sm font-mono text-gray-900 truncate">{model.model_path}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'insights' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-4 flex items-center">
                <Eye className="h-4 w-4 mr-2 text-gray-600" />
                Model Insights
              </h3>
              
              <div className="space-y-4">
                {/* Performance Analysis */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Activity className="h-4 w-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
                    <div>
                      <h4 className="text-sm font-medium text-blue-900">Performance Analysis</h4>
                      <p className="text-sm text-blue-800 mt-1">
                        {problemType === 'CLASSIFICATION' ? (
                          (metrics.accuracy || 0) > 0.9 ? 
                            'Excellent performance! This model shows very high accuracy.' :
                          (metrics.accuracy || 0) > 0.8 ? 
                            'Good performance. Consider fine-tuning for better results.' :
                            'Model performance could be improved. Consider feature engineering or different algorithms.'
                        ) : (
                          (metrics.r2 || 0) > 0.9 ? 
                            'Excellent fit! This model explains most of the variance in your data.' :
                          (metrics.r2 || 0) > 0.7 ? 
                            'Good fit. The model captures most patterns in your data.' :
                            'Model fit could be improved. Consider feature engineering or regularization.'
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Training Speed */}
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Clock className="h-4 w-4 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                    <div>
                      <h4 className="text-sm font-medium text-green-900">Training Efficiency</h4>
                      <p className="text-sm text-green-800 mt-1">
                        {(model.training_time || 0) < 1 ? 
                          'Very fast training time. Great for rapid experimentation.' :
                        (model.training_time || 0) < 10 ? 
                          'Reasonable training time. Good balance of speed and performance.' :
                          'Slower training time. Consider this algorithm for final models only.'
                        }
                      </p>
                    </div>
                  </div>
                </div>

                {/* Interpretability */}
                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Brain className="h-4 w-4 text-purple-600 mt-0.5 mr-2 flex-shrink-0" />
                    <div>
                      <h4 className="text-sm font-medium text-purple-900">Model Interpretability</h4>
                      <p className="text-sm text-purple-800 mt-1">
                        {model.feature_importance ? 
                          'This model provides feature importance scores, making it easy to understand which features drive predictions.' :
                          'This model doesn\'t provide feature importance. Consider using SHAP or LIME for explainability.'
                        }
                      </p>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <Info className="h-4 w-4 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />
                    <div>
                      <h4 className="text-sm font-medium text-yellow-900">Recommendations</h4>
                      <ul className="text-sm text-yellow-800 mt-1 space-y-1">
                        {isBest ? (
                          <li>• This is your best performing model - consider deploying it</li>
                        ) : (
                          <li>• Try hyperparameter tuning to improve performance</li>
                        )}
                        <li>• Cross-validate results to ensure robustness</li>
                        <li>• Test with new data to verify generalization</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelDetails; 