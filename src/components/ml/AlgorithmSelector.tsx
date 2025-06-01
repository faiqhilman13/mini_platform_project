import React, { useState, useEffect } from 'react';
import { 
  Brain,
  Zap,
  Clock,
  CheckCircle2,
  Circle,
  Info,
  AlertCircle,
  TrendingUp,
  BarChart3,
  Activity
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { Algorithm, AlgorithmConfig } from '../../types';
import { getAlgorithmSuggestions } from '../../services/api';

interface AlgorithmSelectorProps {
  problemType: 'CLASSIFICATION' | 'REGRESSION';
  selectedAlgorithms: AlgorithmConfig[];
  onAlgorithmsChange: (algorithms: AlgorithmConfig[]) => void;
  maxSelections?: number;
  showRecommendations?: boolean;
  className?: string;
}

// Sample algorithm data (would be fetched from API)
const getAlgorithmsByType = (problemType: string): Algorithm[] => {
  const classificationAlgorithms: Algorithm[] = [
    {
      name: 'logistic_regression',
      display_name: 'Logistic Regression',
      description: 'Linear classifier using logistic function. Fast and interpretable with good baseline performance.',
      problem_types: ['CLASSIFICATION'],
      hyperparameters: [
        { name: 'C', type: 'float', default: 1.0, min_value: 0.001, max_value: 1000.0, description: 'Regularization strength', required: true },
        { name: 'max_iter', type: 'int', default: 1000, min_value: 100, max_value: 10000, description: 'Maximum iterations', required: true }
      ],
      default_metrics: ['accuracy', 'precision', 'recall', 'f1_score'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical', 'scale_features'],
      min_samples: 50,
      supports_feature_importance: true,
      supports_probabilities: true,
      training_complexity: 'low'
    },
    {
      name: 'random_forest_classifier',
      display_name: 'Random Forest',
      description: 'Ensemble of decision trees with excellent performance and built-in feature importance.',
      problem_types: ['CLASSIFICATION'],
      hyperparameters: [
        { name: 'n_estimators', type: 'int', default: 100, min_value: 10, max_value: 1000, description: 'Number of trees', required: true },
        { name: 'max_depth', type: 'int', default: 10, min_value: 1, max_value: 50, description: 'Maximum tree depth', required: false }
      ],
      default_metrics: ['accuracy', 'precision', 'recall', 'f1_score'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical'],
      min_samples: 20,
      supports_feature_importance: true,
      supports_probabilities: true,
      training_complexity: 'medium'
    },
    {
      name: 'decision_tree_classifier',
      display_name: 'Decision Tree',
      description: 'Tree-based classifier that is highly interpretable and provides clear decision rules.',
      problem_types: ['CLASSIFICATION'],
      hyperparameters: [
        { name: 'max_depth', type: 'int', default: 10, min_value: 1, max_value: 50, description: 'Maximum tree depth', required: true },
        { name: 'min_samples_split', type: 'int', default: 2, min_value: 2, max_value: 100, description: 'Min samples to split', required: true }
      ],
      default_metrics: ['accuracy', 'precision', 'recall', 'f1_score'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical'],
      min_samples: 20,
      supports_feature_importance: true,
      supports_probabilities: true,
      training_complexity: 'low'
    },
    {
      name: 'svm_classifier',
      display_name: 'Support Vector Machine',
      description: 'Powerful classifier that works well with high-dimensional data and complex boundaries.',
      problem_types: ['CLASSIFICATION'],
      hyperparameters: [
        { name: 'C', type: 'float', default: 1.0, min_value: 0.001, max_value: 1000.0, description: 'Regularization parameter', required: true },
        { name: 'kernel', type: 'str', default: 'rbf', allowed_values: ['linear', 'poly', 'rbf', 'sigmoid'], description: 'Kernel type', required: true }
      ],
      default_metrics: ['accuracy', 'precision', 'recall', 'f1_score'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical', 'scale_features'],
      min_samples: 100,
      supports_feature_importance: false,
      supports_probabilities: true,
      training_complexity: 'high'
    },
    {
      name: 'knn_classifier',
      display_name: 'K-Nearest Neighbors',
      description: 'Instance-based classifier that makes predictions based on the closest training examples.',
      problem_types: ['CLASSIFICATION'],
      hyperparameters: [
        { name: 'n_neighbors', type: 'int', default: 5, min_value: 1, max_value: 50, description: 'Number of neighbors', required: true },
        { name: 'weights', type: 'str', default: 'uniform', allowed_values: ['uniform', 'distance'], description: 'Weight function', required: true }
      ],
      default_metrics: ['accuracy', 'precision', 'recall', 'f1_score'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical', 'scale_features'],
      min_samples: 30,
      supports_feature_importance: false,
      supports_probabilities: true,
      training_complexity: 'low'
    }
  ];

  const regressionAlgorithms: Algorithm[] = [
    {
      name: 'linear_regression',
      display_name: 'Linear Regression',
      description: 'Linear relationship modeling. Fast, interpretable, and works well for linear patterns.',
      problem_types: ['REGRESSION'],
      hyperparameters: [
        { name: 'fit_intercept', type: 'bool', default: true, description: 'Whether to fit intercept', required: true }
      ],
      default_metrics: ['r2', 'mae', 'mse', 'rmse'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical', 'scale_features'],
      min_samples: 20,
      supports_feature_importance: true,
      supports_probabilities: false,
      training_complexity: 'low'
    },
    {
      name: 'random_forest_regressor',
      display_name: 'Random Forest',
      description: 'Ensemble of regression trees. Excellent performance and handles non-linear patterns well.',
      problem_types: ['REGRESSION'],
      hyperparameters: [
        { name: 'n_estimators', type: 'int', default: 100, min_value: 10, max_value: 1000, description: 'Number of trees', required: true },
        { name: 'max_depth', type: 'int', default: 10, min_value: 1, max_value: 50, description: 'Maximum tree depth', required: false }
      ],
      default_metrics: ['r2', 'mae', 'mse', 'rmse'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical'],
      min_samples: 20,
      supports_feature_importance: true,
      supports_probabilities: false,
      training_complexity: 'medium'
    },
    {
      name: 'decision_tree_regressor',
      display_name: 'Decision Tree',
      description: 'Tree-based regressor that provides interpretable rules and handles non-linear patterns.',
      problem_types: ['REGRESSION'],
      hyperparameters: [
        { name: 'max_depth', type: 'int', default: 10, min_value: 1, max_value: 50, description: 'Maximum tree depth', required: true },
        { name: 'min_samples_split', type: 'int', default: 2, min_value: 2, max_value: 100, description: 'Min samples to split', required: true }
      ],
      default_metrics: ['r2', 'mae', 'mse', 'rmse'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical'],
      min_samples: 20,
      supports_feature_importance: true,
      supports_probabilities: false,
      training_complexity: 'low'
    },
    {
      name: 'svm_regressor',
      display_name: 'Support Vector Machine',
      description: 'Powerful regressor that works well with high-dimensional data and complex patterns.',
      problem_types: ['REGRESSION'],
      hyperparameters: [
        { name: 'C', type: 'float', default: 1.0, min_value: 0.001, max_value: 1000.0, description: 'Regularization parameter', required: true },
        { name: 'kernel', type: 'str', default: 'rbf', allowed_values: ['linear', 'poly', 'rbf', 'sigmoid'], description: 'Kernel type', required: true }
      ],
      default_metrics: ['r2', 'mae', 'mse', 'rmse'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical', 'scale_features'],
      min_samples: 100,
      supports_feature_importance: false,
      supports_probabilities: false,
      training_complexity: 'high'
    },
    {
      name: 'knn_regressor',
      display_name: 'K-Nearest Neighbors',
      description: 'Instance-based regressor that predicts based on the average of closest training examples.',
      problem_types: ['REGRESSION'],
      hyperparameters: [
        { name: 'n_neighbors', type: 'int', default: 5, min_value: 1, max_value: 50, description: 'Number of neighbors', required: true },
        { name: 'weights', type: 'str', default: 'uniform', allowed_values: ['uniform', 'distance'], description: 'Weight function', required: true }
      ],
      default_metrics: ['r2', 'mae', 'mse', 'rmse'],
      recommended_preprocessing: ['handle_missing', 'encode_categorical', 'scale_features'],
      min_samples: 30,
      supports_feature_importance: false,
      supports_probabilities: false,
      training_complexity: 'low'
    }
  ];

  return problemType === 'CLASSIFICATION' ? classificationAlgorithms : regressionAlgorithms;
};

// Get complexity icon and color
const getComplexityInfo = (complexity: string) => {
  switch (complexity) {
    case 'low':
      return { 
        icon: <Zap className="h-4 w-4" />, 
        color: 'text-green-500', 
        bgColor: 'bg-green-50', 
        label: 'Fast & Simple' 
      };
    case 'medium':
      return { 
        icon: <Activity className="h-4 w-4" />, 
        color: 'text-yellow-500', 
        bgColor: 'bg-yellow-50', 
        label: 'Balanced' 
      };
    case 'high':
      return { 
        icon: <Clock className="h-4 w-4" />, 
        color: 'text-red-500', 
        bgColor: 'bg-red-50', 
        label: 'Complex & Powerful' 
      };
    default:
      return { 
        icon: <Activity className="h-4 w-4" />, 
        color: 'text-gray-500', 
        bgColor: 'bg-gray-50', 
        label: 'Unknown' 
      };
  }
};

const AlgorithmSelector = ({
  problemType,
  selectedAlgorithms,
  onAlgorithmsChange,
  maxSelections = 5,
  showRecommendations = true,
  className
}: AlgorithmSelectorProps) => {
  const [availableAlgorithms, setAvailableAlgorithms] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(false);
  const [showDetails, setShowDetails] = useState<Record<string, boolean>>({});

  // Load algorithms on mount or when problem type changes
  useEffect(() => {
    const loadAlgorithms = async () => {
      setLoading(true);
      try {
        // For now, use static data. In production, this would fetch from API
        const algorithms = getAlgorithmsByType(problemType);
        setAvailableAlgorithms(algorithms);
      } catch (error) {
        console.error('Failed to load algorithms:', error);
      } finally {
        setLoading(false);
      }
    };

    loadAlgorithms();
  }, [problemType]);

  // Check if algorithm is selected
  const isAlgorithmSelected = (algorithmName: string): boolean => {
    return selectedAlgorithms.some(config => config.algorithm_name === algorithmName);
  };

  // Toggle algorithm selection
  const toggleAlgorithm = (algorithm: Algorithm) => {
    const isSelected = isAlgorithmSelected(algorithm.name);
    
    if (isSelected) {
      // Remove algorithm
      const updated = selectedAlgorithms.filter(config => config.algorithm_name !== algorithm.name);
      onAlgorithmsChange(updated);
    } else {
      // Add algorithm (if under limit)
      if (selectedAlgorithms.length >= maxSelections) {
        return; // Don't add if at max
      }
      
      // Create default configuration
      const defaultHyperparams = algorithm.hyperparameters.reduce((acc, param) => {
        acc[param.name] = param.default;
        return acc;
      }, {} as Record<string, any>);
      
      const newConfig: AlgorithmConfig = {
        algorithm_name: algorithm.name,
        hyperparameters: defaultHyperparams,
        is_enabled: true
      };
      
      onAlgorithmsChange([...selectedAlgorithms, newConfig]);
    }
  };

  // Select recommended algorithms
  const selectRecommended = () => {
    const recommended = availableAlgorithms
      .filter(algo => ['random_forest_classifier', 'random_forest_regressor', 'logistic_regression', 'linear_regression'].includes(algo.name))
      .slice(0, 3);
    
    const configs = recommended.map(algo => {
      const defaultHyperparams = algo.hyperparameters.reduce((acc, param) => {
        acc[param.name] = param.default;
        return acc;
      }, {} as Record<string, any>);
      
      return {
        algorithm_name: algo.name,
        hyperparameters: defaultHyperparams,
        is_enabled: true
      };
    });
    
    onAlgorithmsChange(configs);
  };

  // Clear all selections
  const clearAll = () => {
    onAlgorithmsChange([]);
  };

  // Toggle details view for an algorithm
  const toggleDetails = (algorithmName: string) => {
    setShowDetails(prev => ({
      ...prev,
      [algorithmName]: !prev[algorithmName]
    }));
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Brain className="h-5 w-5 mr-2 text-purple-600" />
            Algorithm Selection
          </CardTitle>
          <div className="flex space-x-2">
            {showRecommendations && (
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
              onClick={clearAll}
              className="text-xs"
            >
              Clear All
            </Button>
          </div>
        </div>
        
        <div className="flex items-center justify-between text-sm text-gray-300">
          <span>
            {selectedAlgorithms.length} of {maxSelections} algorithms selected
          </span>
          <span className="capitalize">
            {problemType.toLowerCase()} algorithms
          </span>
        </div>
      </CardHeader>
      
      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600 mr-3"></div>
            <span className="text-gray-500">Loading algorithms...</span>
          </div>
        ) : (
          <div className="space-y-3">
            {availableAlgorithms.map((algorithm) => {
              const isSelected = isAlgorithmSelected(algorithm.name);
              const complexityInfo = getComplexityInfo(algorithm.training_complexity);
              const canSelect = selectedAlgorithms.length < maxSelections || isSelected;
              
              return (
                <motion.div
                  key={algorithm.name}
                  layout
                  className={cn(
                    'border rounded-lg p-4 transition-all cursor-pointer',
                    isSelected 
                      ? 'border-purple-500 bg-purple-50' 
                      : canSelect 
                        ? 'border-gray-200 bg-white hover:border-gray-300' 
                        : 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed'
                  )}
                  onClick={() => canSelect && toggleAlgorithm(algorithm)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1">
                      <div className="mt-1">
                        {isSelected ? (
                          <CheckCircle2 className="h-5 w-5 text-purple-600" />
                        ) : (
                          <Circle className="h-5 w-5 text-gray-400" />
                        )}
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h3 className="font-medium text-gray-900">
                            {algorithm.display_name}
                          </h3>
                          <div className={cn(
                            'flex items-center space-x-1 px-2 py-1 rounded-full text-xs',
                            complexityInfo.bgColor,
                            complexityInfo.color
                          )}>
                            {complexityInfo.icon}
                            <span>{complexityInfo.label}</span>
                          </div>
                        </div>
                        
                        <p className="text-sm text-gray-300 mb-3">
                          {algorithm.description}
                        </p>
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <div className="flex items-center">
                            <BarChart3 className="h-3 w-3 mr-1" />
                            Min: {algorithm.min_samples} samples
                          </div>
                          {algorithm.supports_feature_importance && (
                            <div className="flex items-center">
                              <TrendingUp className="h-3 w-3 mr-1" />
                              Feature importance
                            </div>
                          )}
                          {algorithm.supports_probabilities && problemType === 'CLASSIFICATION' && (
                            <div className="flex items-center">
                              <Activity className="h-3 w-3 mr-1" />
                              Probabilities
                            </div>
                          )}
                        </div>
                        
                        {/* Algorithm Details */}
                        <AnimatePresence>
                          {showDetails[algorithm.name] && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="mt-3 pt-3 border-t border-gray-200"
                            >
                              <div className="space-y-2">
                                <div>
                                  <h4 className="text-xs font-medium text-gray-700 mb-1">
                                    Key Hyperparameters:
                                  </h4>
                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                    {algorithm.hyperparameters.slice(0, 4).map((param) => (
                                      <div key={param.name} className="text-xs">
                                        <span className="font-mono text-gray-300">
                                          {param.name}
                                        </span>
                                        <span className="text-gray-500 ml-1">
                                          (default: {String(param.default)})
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                                
                                <div>
                                  <h4 className="text-xs font-medium text-gray-700 mb-1">
                                    Recommended Preprocessing:
                                  </h4>
                                  <div className="flex flex-wrap gap-1">
                                    {algorithm.recommended_preprocessing.map((step) => (
                                      <span 
                                        key={step}
                                        className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-100 text-blue-700"
                                      >
                                        {step.replace('_', ' ')}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </div>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleDetails(algorithm.name);
                      }}
                      className="text-xs"
                    >
                      <Info className="h-3 w-3" />
                    </Button>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
        
        {/* Selection limit warning */}
        {selectedAlgorithms.length >= maxSelections && (
          <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-start">
              <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-amber-800">Maximum Selection Reached</p>
                <p className="text-xs text-amber-700 mt-1">
                  You've selected the maximum of {maxSelections} algorithms. Deselect one to choose another.
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* Recommendations */}
        {showRecommendations && selectedAlgorithms.length === 0 && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start">
              <Info className="h-4 w-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-blue-800">
                  Get Started with Recommended Algorithms
                </p>
                <p className="text-xs text-blue-700 mt-1">
                  {problemType === 'CLASSIFICATION' 
                    ? 'For classification tasks, Random Forest and Logistic Regression provide excellent starting points.'
                    : 'For regression tasks, Random Forest and Linear Regression offer great baseline performance.'
                  }
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={selectRecommended}
                  className="mt-2 text-xs"
                >
                  <Zap className="h-3 w-3 mr-1" />
                  Select Recommended
                </Button>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AlgorithmSelector;
