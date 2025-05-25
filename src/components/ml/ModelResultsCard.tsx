import React from 'react';
import { Award, AlertCircle, Clock } from 'lucide-react';
import { MLModel } from '../../types';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { cn } from '../../utils/helpers';

interface ModelResultsCardProps {
  model: MLModel;
  isBest?: boolean;
  className?: string;
}

const ModelResultsCard = ({
  model,
  isBest = false,
  className
}: ModelResultsCardProps) => {
  // Helper to format metric values
  const formatMetric = (value: number): string => {
    return value.toFixed(4);
  };

  // Helper to get the primary metric based on model type
  const getPrimaryMetric = (): { name: string; value: number } => {
    const metrics = model.performance_metrics;
    
    // For classification models
    if ('accuracy' in metrics) {
      return { name: 'Accuracy', value: metrics.accuracy };
    }
    // For regression models
    else if ('r2' in metrics) {
      return { name: 'R²', value: metrics.r2 };
    }
    
    // Fallback to the first metric
    const firstKey = Object.keys(metrics)[0];
    return { name: firstKey, value: metrics[firstKey] };
  };

  const primaryMetric = getPrimaryMetric();

  return (
    <Card 
      className={cn(
        'transition-all duration-200',
        isBest && 'ring-2 ring-green-500',
        className
      )}
    >
      <CardHeader className={cn(
        'pb-3',
        isBest ? 'bg-green-50' : ''
      )}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            {model.algorithm_name.split('_').map(word => 
              word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ')}
          </CardTitle>
          {isBest && (
            <div className="flex items-center bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
              <Award className="w-3 h-3 mr-1" />
              Best Model
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-3">
        <div className="flex justify-between items-center mb-4">
          <div className="text-center bg-blue-50 rounded-lg p-3 flex-1 mr-2">
            <p className="text-xs text-gray-500 mb-1">{primaryMetric.name}</p>
            <p className="text-xl font-bold text-blue-700">
              {formatMetric(primaryMetric.value)}
            </p>
          </div>
          <div className="text-center bg-gray-50 rounded-lg p-3 flex-1 ml-2">
            <p className="text-xs text-gray-500 mb-1">Training Time</p>
            <p className="flex items-center justify-center text-lg font-medium text-gray-700">
              <Clock className="w-3 h-3 mr-1" />
              {model.training_time.toFixed(2)}s
            </p>
          </div>
        </div>

        <h4 className="text-sm font-medium text-gray-700 mb-2">Performance Metrics:</h4>
        <div className="grid grid-cols-2 gap-2 mb-4">
          {Object.entries(model.performance_metrics)
            .filter(([key]) => key !== primaryMetric.name.toLowerCase())
            .slice(0, 4)
            .map(([key, value]) => (
              <div key={key} className="bg-gray-50 p-2 rounded">
                <p className="text-xs text-gray-500 capitalize">{key}</p>
                <p className="text-sm font-medium">{formatMetric(value as number)}</p>
              </div>
            ))}
        </div>

        {model.feature_importance && Object.keys(model.feature_importance).length > 0 && (
          <>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Feature Importance:</h4>
            <div className="space-y-2">
              {Object.entries(model.feature_importance)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([feature, importance]) => (
                  <div key={feature} className="flex items-center">
                    <div className="text-xs text-gray-600 w-1/3 truncate" title={feature}>
                      {feature}
                    </div>
                    <div className="w-2/3 flex items-center">
                      <div 
                        className="bg-blue-500 h-2 rounded"
                        style={{ width: `${Math.min(100, importance * 100)}%` }}
                      ></div>
                      <span className="ml-2 text-xs text-gray-500">
                        {(importance * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
            </div>
          </>
        )}

        <div className="mt-4 text-xs text-gray-500 flex items-center">
          <AlertCircle className="w-3 h-3 mr-1" />
          <span>Model ID: {model.model_id.substring(0, 8)}...</span>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelResultsCard;