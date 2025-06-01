import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  BarChart3, 
  Trophy, 
  Clock, 
  Target, 
  Brain,
  Search,
  Filter,
  RefreshCw,
  AlertCircle,
  Eye,
  Calendar,
  TrendingUp,
  CheckCircle,
  XCircle,
  Activity
} from 'lucide-react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import Button from '../components/ui/Button';
import { getPipelineRuns } from '../services/api';
import { PipelineRun, MLPipelineRun } from '../types';
import { cn } from '../utils/helpers';

const ResultsListPage = () => {
  const navigate = useNavigate();
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRun[]>([]);
  const [filteredRuns, setFilteredRuns] = useState<PipelineRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'COMPLETED' | 'PROCESSING' | 'FAILED'>('all');

  const fetchPipelineRuns = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const runs = await getPipelineRuns();
      // Filter for ML training runs only
      const mlRuns = runs.filter(run => run.pipeline_type === 'ML_TRAINING');
      setPipelineRuns(mlRuns);
      setFilteredRuns(mlRuns);
    } catch (err) {
      setError('Failed to load results. Please try again.');
      console.error('Error fetching pipeline runs:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPipelineRuns();
  }, [fetchPipelineRuns]);

  // Filter and search runs
  useEffect(() => {
    let filtered = pipelineRuns;

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(run => run.status === statusFilter);
    }

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(run =>
        run.run_uuid.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (run as MLPipelineRun).target_variable?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredRuns(filtered);
  }, [pipelineRuns, statusFilter, searchTerm]);

  const handleViewResults = (run: PipelineRun) => {
    navigate(`/ml/results/${run.run_uuid}`);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'PROCESSING':
        return <Activity className="h-4 w-4 text-blue-600 animate-pulse" />;
      case 'FAILED':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-300" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'PROCESSING':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'FAILED':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRunStats = () => {
    const stats = {
      total: pipelineRuns.length,
      completed: pipelineRuns.filter(run => run.status === 'COMPLETED').length,
      processing: pipelineRuns.filter(run => run.status === 'PROCESSING').length,
      failed: pipelineRuns.filter(run => run.status === 'FAILED').length,
    };
    return stats;
  };

  const stats = getRunStats();

  if (isLoading) {
    return (
      <div className="px-4 py-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
            <p className="text-gray-300">Loading results...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-black dark:text-white flex items-center">
              <BarChart3 className="h-8 w-8 text-purple-600 mr-3" />
              ML Training Results
            </h1>
            <p className="text-black dark:text-gray-300 mt-2">
              View and analyze your machine learning model training results
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              onClick={fetchPipelineRuns}
              className="text-sm"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
            <Button
              onClick={() => navigate('/ml')}
              className="text-sm"
            >
              <Brain className="h-4 w-4 mr-1" />
              Start Training
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <BarChart3 className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-black dark:text-gray-300">Total Runs</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.total}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-black dark:text-gray-300">Completed</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.completed}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Activity className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-black dark:text-gray-300">Processing</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.processing}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-red-100 rounded-lg">
                  <XCircle className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-black dark:text-gray-300">Failed</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.failed}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters and Search */}
        <Card>
          <CardContent className="p-6">
            <div className="flex flex-col sm:flex-row gap-4">
              {/* Search */}
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search by run ID or target variable..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>

              {/* Status Filter */}
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-gray-500" />
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value as any)}
                  className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-700"
                >
                  <option value="all">All Status</option>
                  <option value="COMPLETED">Completed</option>
                  <option value="PROCESSING">Processing</option>
                  <option value="FAILED">Failed</option>
                </select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Results List */}
        {error ? (
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center text-red-500">
                <AlertCircle className="h-5 w-5 mr-2" />
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        ) : filteredRuns.length === 0 ? (
          <Card>
            <CardContent className="p-6">
              <div className="text-center py-8">
                <BarChart3 className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-black dark:text-white mb-2">
                  {searchTerm || statusFilter !== 'all' ? 'No results found' : 'No training runs yet'}
                </h3>
                <p className="text-black dark:text-gray-300 mb-4">
                  {searchTerm || statusFilter !== 'all'
                    ? 'Try adjusting your search or filter criteria.'
                    : 'Start your first ML training to see results here.'
                  }
                </p>
                {!searchTerm && statusFilter === 'all' && (
                  <Button onClick={() => navigate('/ml')}>
                    <Brain className="h-4 w-4 mr-1" />
                    Start Training
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {filteredRuns.map((run, index) => {
              const mlRun = run as MLPipelineRun;
              return (
                <motion.div
                  key={run.run_uuid}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <div className="hover:shadow-md transition-shadow cursor-pointer" onClick={() => handleViewResults(run)}>
                    <Card>
                      <CardContent className="p-6">
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-3 mb-2">
                              <h3 className="text-lg font-medium text-black dark:text-white">
                                Training Run
                              </h3>
                              <span className={cn(
                                'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border',
                                getStatusColor(run.status)
                              )}>
                                {getStatusIcon(run.status)}
                                <span className="ml-1">{run.status}</span>
                              </span>
                            </div>
                            
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-black dark:text-gray-300">
                              <div className="flex items-center">
                                <Target className="h-4 w-4 mr-2 text-gray-400" />
                                <span className="font-medium">Target:</span>
                                <span className="ml-1">{mlRun.target_variable || 'N/A'}</span>
                              </div>
                              
                              <div className="flex items-center">
                                <Brain className="h-4 w-4 mr-2 text-gray-400" />
                                <span className="font-medium">Type:</span>
                                <span className="ml-1 capitalize">
                                  {mlRun.problem_type?.toLowerCase() || 'N/A'}
                                </span>
                              </div>
                              
                              <div className="flex items-center">
                                <Calendar className="h-4 w-4 mr-2 text-gray-400" />
                                <span className="font-medium">Started:</span>
                                <span className="ml-1">
                                  {new Date(run.created_at).toLocaleDateString()}
                                </span>
                              </div>
                            </div>
                            
                            <div className="mt-3 text-xs text-gray-500">
                              Run ID: {run.run_uuid.slice(0, 8)}...
                            </div>
                          </div>
                          
                          <div className="ml-6 flex items-center space-x-3">
                            {run.status === 'COMPLETED' && mlRun.best_model_id && (
                              <div className="text-center">
                                <Trophy className="h-5 w-5 text-yellow-600 mx-auto mb-1" />
                                <span className="text-xs text-black dark:text-gray-300">Best Model</span>
                              </div>
                            )}
                            
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleViewResults(run);
                              }}
                              className="text-xs"
                            >
                              <Eye className="h-3 w-3 mr-1" />
                              View Details
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}

        {/* Quick Actions */}
        <Card className="bg-gradient-to-r from-purple-50 to-blue-50 border-purple-200">
          <CardContent className="p-6">
            <div className="text-center">
              <TrendingUp className="h-12 w-12 text-purple-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-black dark:text-white mb-2">Ready for More?</h3>
              <p className="text-black dark:text-gray-300 mb-4">
                Start new training runs or explore your existing datasets for more insights.
              </p>
              <div className="flex justify-center space-x-3">
                <Button onClick={() => navigate('/ml')}>
                  <Brain className="h-4 w-4 mr-1" />
                  Start Training
                </Button>
                <Button variant="outline" onClick={() => navigate('/files')}>
                  <BarChart3 className="h-4 w-4 mr-1" />
                  Browse Files
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ResultsListPage; 
