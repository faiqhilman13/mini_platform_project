import React, { useState, useEffect } from 'react';
import { 
  Settings,
  RotateCcw,
  Info,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  Gauge,
  Sliders
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { Algorithm, AlgorithmConfig, AlgorithmHyperParameter } from '../../types';

interface HyperparameterConfigProps {
  algorithms: Algorithm[];
  selectedConfigs: AlgorithmConfig[];
  onConfigsChange: (configs: AlgorithmConfig[]) => void;
  mode?: 'basic' | 'advanced';
  showValidation?: boolean;
  className?: string;
}

interface ParameterInputProps {
  parameter: AlgorithmHyperParameter;
  value: any;
  onChange: (value: any) => void;
  error?: string;
  className?: string;
}

// Individual parameter input component
const ParameterInput = ({ parameter, value, onChange, error, className }: ParameterInputProps) => {
  const [localValue, setLocalValue] = useState(value);
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (newValue: any) => {
    setLocalValue(newValue);
    
    // Type conversion and validation
    try {
      let convertedValue = newValue;
      
      if (parameter.type === 'int') {
        convertedValue = parseInt(newValue, 10);
        if (isNaN(convertedValue)) return;
      } else if (parameter.type === 'float') {
        convertedValue = parseFloat(newValue);
        if (isNaN(convertedValue)) return;
      } else if (parameter.type === 'bool') {
        convertedValue = newValue === 'true' || newValue === true;
      }
      
      onChange(convertedValue);
    } catch (e) {
      // Keep local value but don't propagate invalid values
    }
  };

  const renderInput = () => {
    switch (parameter.type) {
      case 'int':
      case 'float':
        return (
          <div className="space-y-1">
            <input
              type="number"
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              min={parameter.min_value}
              max={parameter.max_value}
              step={parameter.type === 'float' ? 0.01 : 1}
              className={cn(
                'w-full px-3 py-2 border rounded-md text-sm transition-colors',
                error 
                  ? 'border-red-300 bg-red-50 focus:border-red-500 focus:ring-red-500' 
                  : focused
                    ? 'border-blue-500 focus:ring-blue-500'
                    : 'border-gray-300 focus:border-blue-500',
                'focus:outline-none focus:ring-1'
              )}
            />
            {(parameter.min_value !== undefined || parameter.max_value !== undefined) && (
              <div className="flex justify-between text-xs text-gray-500">
                <span>Min: {parameter.min_value ?? 'None'}</span>
                <span>Max: {parameter.max_value ?? 'None'}</span>
              </div>
            )}
          </div>
        );
      
      case 'bool':
        return (
          <div className="flex items-center space-x-3">
            <label className="flex items-center">
              <input
                type="radio"
                name={parameter.name}
                value="true"
                checked={localValue === true}
                onChange={() => handleChange(true)}
                className="mr-2"
              />
              True
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name={parameter.name}
                value="false"
                checked={localValue === false}
                onChange={() => handleChange(false)}
                className="mr-2"
              />
              False
            </label>
          </div>
        );
      
      case 'str':
        if (parameter.allowed_values && parameter.allowed_values.length > 0) {
          return (
            <select
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              className={cn(
                'w-full px-3 py-2 border rounded-md text-sm transition-colors',
                error 
                  ? 'border-red-300 bg-red-50 focus:border-red-500 focus:ring-red-500' 
                  : focused
                    ? 'border-blue-500 focus:ring-blue-500'
                    : 'border-gray-300 focus:border-blue-500',
                'focus:outline-none focus:ring-1'
              )}
            >
              {parameter.allowed_values.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          );
        } else {
          return (
            <input
              type="text"
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              className={cn(
                'w-full px-3 py-2 border rounded-md text-sm transition-colors',
                error 
                  ? 'border-red-300 bg-red-50 focus:border-red-500 focus:ring-red-500' 
                  : focused
                    ? 'border-blue-500 focus:ring-blue-500'
                    : 'border-gray-300 focus:border-blue-500',
                'focus:outline-none focus:ring-1'
              )}
            />
          );
        }
      
      default:
        return (
          <input
            type="text"
            value={localValue}
            onChange={(e) => handleChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
        );
    }
  };

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium text-gray-700">
          {parameter.name}
          {parameter.required && <span className="text-red-500 ml-1">*</span>}
        </label>
        {parameter.description && (
          <div className="group relative">
            <Info className="h-4 w-4 text-gray-400 hover:text-gray-600 cursor-help" />
            <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block z-10">
              <div className="bg-gray-900 text-white text-xs rounded py-1 px-2 max-w-xs">
                {parameter.description}
              </div>
            </div>
          </div>
        )}
      </div>
      
      {renderInput()}
      
      {error && (
        <div className="mt-1 flex items-center text-xs text-red-600">
          <AlertCircle className="h-3 w-3 mr-1" />
          {error}
        </div>
      )}
    </div>
  );
};

const HyperparameterConfig = ({
  algorithms,
  selectedConfigs,
  onConfigsChange,
  mode = 'basic',
  showValidation = true,
  className
}: HyperparameterConfigProps) => {
  const [currentMode, setCurrentMode] = useState<'basic' | 'advanced'>(mode);
  const [expandedAlgorithms, setExpandedAlgorithms] = useState<Set<string>>(new Set());
  const [validationErrors, setValidationErrors] = useState<Record<string, Record<string, string>>>({});

  // Validate a parameter value
  const validateParameter = (parameter: AlgorithmHyperParameter, value: any): string | null => {
    if (parameter.required && (value === null || value === undefined || value === '')) {
      return `${parameter.name} is required`;
    }

    if (value === null || value === undefined || value === '') {
      return null; // Optional parameter with no value is valid
    }

    // Type validation
    if (parameter.type === 'int' && (!Number.isInteger(Number(value)) || isNaN(Number(value)))) {
      return `${parameter.name} must be an integer`;
    }

    if (parameter.type === 'float' && isNaN(Number(value))) {
      return `${parameter.name} must be a number`;
    }

    // Range validation
    if ((parameter.type === 'int' || parameter.type === 'float') && typeof value === 'number') {
      if (parameter.min_value !== undefined && value < parameter.min_value) {
        return `${parameter.name} must be at least ${parameter.min_value}`;
      }
      if (parameter.max_value !== undefined && value > parameter.max_value) {
        return `${parameter.name} must be at most ${parameter.max_value}`;
      }
    }

    // Allowed values validation
    if (parameter.allowed_values && !parameter.allowed_values.includes(value)) {
      return `${parameter.name} must be one of: ${parameter.allowed_values.join(', ')}`;
    }

    return null;
  };

  // Validate all configurations
  const validateConfigurations = () => {
    const errors: Record<string, Record<string, string>> = {};
    
    selectedConfigs.forEach(config => {
      const algorithm = algorithms.find(algo => algo.name === config.algorithm_name);
      if (!algorithm) return;
      
      const algorithmErrors: Record<string, string> = {};
      
      algorithm.hyperparameters.forEach(param => {
        const value = config.hyperparameters[param.name];
        const error = validateParameter(param, value);
        if (error) {
          algorithmErrors[param.name] = error;
        }
      });
      
      if (Object.keys(algorithmErrors).length > 0) {
        errors[config.algorithm_name] = algorithmErrors;
      }
    });
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Update hyperparameter value
  const updateHyperparameter = (algorithmName: string, paramName: string, value: any) => {
    const updatedConfigs = selectedConfigs.map(config => {
      if (config.algorithm_name === algorithmName) {
        return {
          ...config,
          hyperparameters: {
            ...config.hyperparameters,
            [paramName]: value
          }
        };
      }
      return config;
    });
    
    onConfigsChange(updatedConfigs);
  };

  // Reset to defaults for an algorithm
  const resetToDefaults = (algorithmName: string) => {
    const algorithm = algorithms.find(algo => algo.name === algorithmName);
    if (!algorithm) return;
    
    const defaultHyperparams = algorithm.hyperparameters.reduce((acc, param) => {
      acc[param.name] = param.default;
      return acc;
    }, {} as Record<string, any>);
    
    const updatedConfigs = selectedConfigs.map(config => {
      if (config.algorithm_name === algorithmName) {
        return {
          ...config,
          hyperparameters: defaultHyperparams
        };
      }
      return config;
    });
    
    onConfigsChange(updatedConfigs);
  };

  // Toggle algorithm expansion
  const toggleExpanded = (algorithmName: string) => {
    const newExpanded = new Set(expandedAlgorithms);
    if (newExpanded.has(algorithmName)) {
      newExpanded.delete(algorithmName);
    } else {
      newExpanded.add(algorithmName);
    }
    setExpandedAlgorithms(newExpanded);
  };

  // Filter parameters based on mode
  const getVisibleParameters = (parameters: AlgorithmHyperParameter[]) => {
    if (currentMode === 'basic') {
      // In basic mode, show only the most important parameters
      return parameters.filter(param => 
        param.required || 
        ['n_estimators', 'max_depth', 'C', 'n_neighbors', 'learning_rate'].includes(param.name)
      );
    }
    return parameters; // Advanced mode shows all parameters
  };

  // Validate on config changes
  useEffect(() => {
    if (showValidation) {
      validateConfigurations();
    }
  }, [selectedConfigs, algorithms, showValidation]);

  if (selectedConfigs.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Settings className="h-5 w-5 mr-2 text-orange-600" />
            Hyperparameter Configuration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <Sliders className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <p>Select algorithms to configure their hyperparameters</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Settings className="h-5 w-5 mr-2 text-orange-600" />
            Hyperparameter Configuration
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            {/* Mode Toggle */}
            <div className="flex items-center bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setCurrentMode('basic')}
                className={cn(
                  'px-3 py-1 text-xs rounded-md transition-colors',
                  currentMode === 'basic' 
                    ? 'bg-white text-gray-900 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-900'
                )}
              >
                Basic
              </button>
              <button
                onClick={() => setCurrentMode('advanced')}
                className={cn(
                  'px-3 py-1 text-xs rounded-md transition-colors',
                  currentMode === 'advanced' 
                    ? 'bg-white text-gray-900 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-900'
                )}
              >
                Advanced
              </button>
            </div>
          </div>
        </div>
        
        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>
            Configuring {selectedConfigs.length} algorithm{selectedConfigs.length !== 1 ? 's' : ''}
          </span>
          <span className="capitalize">
            {currentMode} mode
          </span>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-4">
          {selectedConfigs.map((config) => {
            const algorithm = algorithms.find(algo => algo.name === config.algorithm_name);
            if (!algorithm) return null;
            
            const isExpanded = expandedAlgorithms.has(config.algorithm_name);
            const visibleParameters = getVisibleParameters(algorithm.hyperparameters);
            const hasErrors = validationErrors[config.algorithm_name];
            const errorCount = hasErrors ? Object.keys(hasErrors).length : 0;
            
            return (
              <motion.div
                key={config.algorithm_name}
                layout
                className="border border-gray-200 rounded-lg"
              >
                {/* Algorithm Header */}
                <div 
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50"
                  onClick={() => toggleExpanded(config.algorithm_name)}
                >
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center">
                      {isExpanded ? (
                        <ChevronUp className="h-4 w-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-gray-500" />
                      )}
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-gray-900">
                        {algorithm.display_name}
                      </h3>
                      <p className="text-sm text-gray-500">
                        {visibleParameters.length} parameter{visibleParameters.length !== 1 ? 's' : ''} 
                        {currentMode === 'basic' && algorithm.hyperparameters.length > visibleParameters.length && 
                          ` (${algorithm.hyperparameters.length - visibleParameters.length} hidden)`
                        }
                      </p>
                    </div>
                    
                    {showValidation && (
                      <div className="flex items-center">
                        {errorCount > 0 ? (
                          <div className="flex items-center text-red-600">
                            <AlertCircle className="h-4 w-4 mr-1" />
                            <span className="text-xs">{errorCount} error{errorCount !== 1 ? 's' : ''}</span>
                          </div>
                        ) : (
                          <div className="flex items-center text-green-600">
                            <CheckCircle className="h-4 w-4 mr-1" />
                            <span className="text-xs">Valid</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      resetToDefaults(config.algorithm_name);
                    }}
                    className="text-xs"
                  >
                    <RotateCcw className="h-3 w-3 mr-1" />
                    Reset
                  </Button>
                </div>
                
                {/* Parameters */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="border-t border-gray-200"
                    >
                      <div className="p-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {visibleParameters.map((parameter) => (
                            <ParameterInput
                              key={parameter.name}
                              parameter={parameter}
                              value={config.hyperparameters[parameter.name]}
                              onChange={(value) => updateHyperparameter(config.algorithm_name, parameter.name, value)}
                              error={hasErrors?.[parameter.name]}
                            />
                          ))}
                        </div>
                        
                        {currentMode === 'basic' && algorithm.hyperparameters.length > visibleParameters.length && (
                          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                            <div className="flex items-center">
                              <Info className="h-4 w-4 text-blue-600 mr-2" />
                              <p className="text-sm text-blue-800">
                                {algorithm.hyperparameters.length - visibleParameters.length} additional parameters 
                                available in Advanced mode
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </div>
        
        {/* Summary */}
        {showValidation && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Gauge className="h-4 w-4 text-gray-600 mr-2" />
                <span className="text-sm font-medium text-gray-700">Configuration Status</span>
              </div>
              
              <div className="text-sm text-gray-600">
                {Object.keys(validationErrors).length === 0 ? (
                  <span className="text-green-600 flex items-center">
                    <CheckCircle className="h-4 w-4 mr-1" />
                    All algorithms configured correctly
                  </span>
                ) : (
                  <span className="text-red-600 flex items-center">
                    <AlertCircle className="h-4 w-4 mr-1" />
                    {Object.keys(validationErrors).length} algorithm{Object.keys(validationErrors).length !== 1 ? 's' : ''} with errors
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default HyperparameterConfig;