import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Trophy,
  Download,
  BarChart3,
  Clock,
  Target,
  TrendingUp,
  Eye,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Brain,
  Zap,
  Activity
} from 'lucide-react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import Button from '../components/ui/Button';
import { cn } from '../utils/helpers';
import { MLPipelineRun, MLModel } from '../types';
import { getMLPipelineStatus, getMLModels } from '../services/api';
import ModelDetails from '../components/ml/ModelDetails';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

const MetricCard = ({ title, value, subtitle, icon, color, trend }: MetricCardProps) => (
  <Card className="relative overflow-hidden">
    <CardContent className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
        <div className={cn('p-3 rounded-full', color)}>
          {icon}
        </div>
      </div>
      
      {trend && (
        <div className="mt-4 flex items-center">
          <TrendingUp className={cn(
            'h-4 w-4 mr-1',
            trend.isPositive ? 'text-green-600' : 'text-red-600',
            !trend.isPositive && 'rotate-180'
          )} />
          <span className={cn(
            'text-sm font-medium',
            trend.isPositive ? 'text-green-600' : 'text-red-600'
          )}>
            {Math.abs(trend.value).toFixed(1)}%
          </span>
          <span className="text-sm text-gray-500 ml-1">vs baseline</span>
        </div>
      )}
    </CardContent>
  </Card>
);

interface ModelRowProps {
  model: MLModel;
  isBest: boolean;
  onViewDetails: (model: MLModel) => void;
  problemType: string;
}

const ModelRow = ({ model, isBest, onViewDetails, problemType }: ModelRowProps) => {
  const metrics = model.performance_metrics;
  
  const getPrimaryMetric = () => {
    if (problemType === 'CLASSIFICATION') {
      return {
        name: 'Accuracy',
        value: metrics.accuracy?.toFixed(4) || 'N/A',
        secondaryName: 'F1-Score',
        secondaryValue: metrics.f1_score?.toFixed(4) || 'N/A'
      };
    } else {
      return {
        name: 'R² Score',
        value: metrics.r2?.toFixed(4) || 'N/A',
        secondaryName: 'RMSE',
        secondaryValue: metrics.rmse?.toFixed(2) || 'N/A'
      };
    }
  };

  const primaryMetric = getPrimaryMetric();

  return (
    <motion.tr
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'border-b border-gray-200 hover:bg-gray-50 transition-colors',
        isBest && 'bg-yellow-50 border-yellow-200'
      )}
    >
      <td className="px-6 py-4">
        <div className="flex items-center">
          {isBest && (
            <Trophy className="h-4 w-4 text-yellow-600 mr-2" />
          )}
          <div>
            <div className="font-medium text-gray-900">
              {model.algorithm_name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </div>
            <div className="text-sm text-gray-500">
              {Object.keys(model.hyperparameters).length} hyperparameters
            </div>
          </div>
        </div>
      </td>
      
      <td className="px-6 py-4">
        <div className="text-sm font-medium text-gray-900">
          {primaryMetric.value}
        </div>
        <div className="text-xs text-gray-500">
          {primaryMetric.name}
        </div>
      </td>
      
      <td className="px-6 py-4">
        <div className="text-sm font-medium text-gray-900">
          {primaryMetric.secondaryValue}
        </div>
        <div className="text-xs text-gray-500">
          {primaryMetric.secondaryName}
        </div>
      </td>
      
      <td className="px-6 py-4">
        <div className="text-sm text-gray-900">
          {model.training_time ? `${model.training_time.toFixed(2)}s` : 'N/A'}
        </div>
      </td>
      
      <td className="px-6 py-4">
        <div className="flex items-center space-x-2">
          {model.feature_importance && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
              <BarChart3 className="h-3 w-3 mr-1" />
              Feature Importance
            </span>
          )}
        </div>
      </td>
      
      <td className="px-6 py-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewDetails(model)}
          className="text-xs"
        >
          <Eye className="h-3 w-3 mr-1" />
          Details
        </Button>
      </td>
    </motion.tr>
  );
};

const MLResultsPage = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  
  const [pipelineRun, setPipelineRun] = useState<MLPipelineRun | null>(null);
  const [models, setModels] = useState<MLModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);

  useEffect(() => {
    if (runId) {
      loadResults();
    }
  }, [runId]);

  const loadResults = async () => {
    if (!runId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const [pipelineData, modelsData] = await Promise.all([
        getMLPipelineStatus(runId),
        getMLModels(runId)
      ]);
      
      setPipelineRun(pipelineData);
      setModels(modelsData);
    } catch (err) {
      setError('Failed to load results. Please try again.');
      console.error('Error loading results:', err);
    } finally {
      setLoading(false);
    }
  };

  const getBestModel = () => {
    if (!pipelineRun?.best_model_id) return null;
    return models.find(model => model.model_id === pipelineRun.best_model_id);
  };

  const getOverallStats = () => {
    if (models.length === 0) return null;
    
    const bestModel = getBestModel();
    const totalTrainingTime = models.reduce((sum, model) => sum + (model.training_time || 0), 0);
    
    let primaryMetricValue = 'N/A';
    let primaryMetricName = '';
    
    if (bestModel) {
      if (pipelineRun?.problem_type === 'CLASSIFICATION') {
        primaryMetricValue = bestModel.performance_metrics.accuracy?.toFixed(4) || 'N/A';
        primaryMetricName = 'Accuracy';
      } else {
        primaryMetricValue = bestModel.performance_metrics.r2?.toFixed(4) || 'N/A';
        primaryMetricName = 'R² Score';
      }
    }
    
    return {
      bestScore: primaryMetricValue,
      bestScoreName: primaryMetricName,
      totalModels: models.length,
      totalTime: totalTrainingTime.toFixed(2),
      bestAlgorithm: bestModel?.algorithm_name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'N/A'
    };
  };

    const handleViewDetails = (model: MLModel) => {    setSelectedModel(model);  };

  const handleExport = () => {
    // TODO: Implement export functionality
    console.log('Exporting results...');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Error Loading Results</h3>
            <p className="text-gray-600 mb-4">{error}</p>
            <div className="space-x-3">
              <Button onClick={loadResults} className="text-sm">
                <RefreshCw className="h-4 w-4 mr-1" />
                Retry
              </Button>
              <Button variant="outline" onClick={() => navigate(-1)} className="text-sm">
                Go Back
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!pipelineRun) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 text-center">
            <AlertCircle className="h-12 w-12 text-gray-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Found</h3>
            <p className="text-gray-600 mb-4">No results found for this pipeline run.</p>
            <Button variant="outline" onClick={() => navigate(-1)} className="text-sm">
              Go Back
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const stats = getOverallStats();
  const bestModel = getBestModel();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center">
              <Brain className="h-8 w-8 text-purple-600 mr-3" />
              ML Training Results
            </h1>
            <p className="text-gray-600 mt-2">
              {pipelineRun.problem_type ? 
                pipelineRun.problem_type.toLowerCase().replace(/^\w/, c => c.toUpperCase()) : 
                'Unknown Problem Type'
              } • 
              Run started {new Date(pipelineRun.created_at || Date.now()).toLocaleDateString()}
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              onClick={handleExport}
              className="text-sm"
            >
              <Download className="h-4 w-4 mr-1" />
              Export Results
            </Button>
            <Button
              variant="outline"
              onClick={loadResults}
              className="text-sm"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Status Banner */}
        <div className={cn(
          'mb-8 p-4 rounded-lg border',
          pipelineRun.status === 'COMPLETED' 
            ? 'bg-green-50 border-green-200 text-green-800'
            : pipelineRun.status === 'FAILED'
              ? 'bg-red-50 border-red-200 text-red-800'
              : 'bg-yellow-50 border-yellow-200 text-yellow-800'
        )}>
          <div className="flex items-center">
            {pipelineRun.status === 'COMPLETED' ? (
              <CheckCircle className="h-5 w-5 mr-2" />
            ) : pipelineRun.status === 'FAILED' ? (
              <AlertCircle className="h-5 w-5 mr-2" />
            ) : (
              <Activity className="h-5 w-5 mr-2 animate-pulse" />
            )}
            <span className="font-medium">
              Pipeline {pipelineRun.status.toLowerCase()}
            </span>
            {pipelineRun.status === 'COMPLETED' && stats && (
              <span className="ml-2">
                • {stats.totalModels} models trained in {stats.totalTime}s
              </span>
            )}
          </div>
        </div>

        {/* Overview Metrics */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <MetricCard
              title="Best Performance"
              value={stats.bestScore}
              subtitle={stats.bestScoreName}
              icon={<Trophy className="h-6 w-6 text-white" />}
              color="bg-yellow-500"
            />
            
            <MetricCard
              title="Models Trained"
              value={stats.totalModels}
              subtitle="Algorithms compared"
              icon={<Brain className="h-6 w-6 text-white" />}
              color="bg-purple-500"
            />
            
            <MetricCard
              title="Training Time"
              value={`${stats.totalTime}s`}
              subtitle="Total duration"
              icon={<Clock className="h-6 w-6 text-white" />}
              color="bg-blue-500"
            />
            
            <MetricCard
              title="Best Algorithm"
              value={stats.bestAlgorithm}
              subtitle="Top performer"
              icon={<Zap className="h-6 w-6 text-white" />}
              color="bg-green-500"
            />
          </div>
        )}

        {/* Models Comparison Table */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-purple-600" />
              Model Performance Comparison
            </CardTitle>
          </CardHeader>
          <CardContent>
            {models.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Algorithm
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Primary Metric
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Secondary Metric
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Training Time
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Features
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {models
                      .sort((a, b) => {
                        // Sort by best model first, then by primary metric
                        if (bestModel && a.model_id === bestModel.model_id) return -1;
                        if (bestModel && b.model_id === bestModel.model_id) return 1;
                        
                        const metricA = pipelineRun?.problem_type === 'CLASSIFICATION' 
                          ? a.performance_metrics?.accuracy || 0
                          : a.performance_metrics?.r2 || 0;
                        const metricB = pipelineRun?.problem_type === 'CLASSIFICATION'
                          ? b.performance_metrics?.accuracy || 0
                          : b.performance_metrics?.r2 || 0;
                        
                        return metricB - metricA;
                      })
                      .map((model) => (
                        <ModelRow
                          key={model.model_id}
                          model={model}
                          isBest={bestModel?.model_id === model.model_id}
                          onViewDetails={handleViewDetails}
                          problemType={pipelineRun.problem_type || 'unknown'}
                        />
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Brain className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <p>No models found for this pipeline run.</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Details Modal/Section */}        {selectedModel && (          <ModelDetails            model={selectedModel}            problemType={pipelineRun.problem_type || 'unknown'}            isBest={selectedModel.model_id === pipelineRun.best_model_id}            className="mb-8"          />        )}        {/* Pipeline Configuration Summary */}        <Card>          <CardHeader>            <CardTitle className="flex items-center">              <Target className="h-5 w-5 mr-2 text-gray-600" />              Pipeline Configuration            </CardTitle>          </CardHeader>          <CardContent>            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">              <div>                <h4 className="text-sm font-medium text-gray-900 mb-2">Problem Type</h4>                <p className="text-sm text-gray-600 capitalize">                  {pipelineRun.problem_type ? pipelineRun.problem_type.toLowerCase() : 'Unknown'}                </p>              </div>                            <div>                <h4 className="text-sm font-medium text-gray-900 mb-2">Target Variable</h4>                <p className="text-sm text-gray-600">                  {pipelineRun.target_variable || 'Not specified'}                </p>              </div>                            <div>                <h4 className="text-sm font-medium text-gray-900 mb-2">Algorithms</h4>                <p className="text-sm text-gray-600">                  {models.length} configured                </p>              </div>                            {pipelineRun.preprocessing_config && (                <>                  <div>                    <h4 className="text-sm font-medium text-gray-900 mb-2">Missing Values</h4>                    <p className="text-sm text-gray-600 capitalize">                      {pipelineRun.preprocessing_config.missing_strategy || 'Not specified'}                    </p>                  </div>                                    <div>                    <h4 className="text-sm font-medium text-gray-900 mb-2">Feature Scaling</h4>                    <p className="text-sm text-gray-600 capitalize">                      {pipelineRun.preprocessing_config.scaling_strategy || 'Not specified'}                    </p>                  </div>                                    <div>                    <h4 className="text-sm font-medium text-gray-900 mb-2">Test Split</h4>                    <p className="text-sm text-gray-600">                      {pipelineRun.preprocessing_config.test_size ?                         `${(pipelineRun.preprocessing_config.test_size * 100).toFixed(0)}%` :                         'Not specified'                      }                    </p>                  </div>                </>              )}            </div>          </CardContent>        </Card>
      </div>
    </div>
  );
};

export default MLResultsPage;