import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, 
  Database, 
  Upload, 
  Play, 
  BarChart3, 
  Target,
  Zap,
  TrendingUp,
  RefreshCw,
  AlertCircle,
  FileText,
  Settings
} from 'lucide-react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import Button from '../components/ui/Button';
import FileCard from '../components/file/FileCard';
import { getUploadedFiles } from '../services/api';
import { UploadedFile } from '../types';

const MLTrainingPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchFiles = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const uploadedFiles = await getUploadedFiles();
      // Filter for datasets only
      const datasets = uploadedFiles.filter(file => ['csv', 'xlsx'].includes(file.file_type));
      setFiles(datasets);
    } catch (err) {
      setError('Failed to load datasets. Please try again.');
      console.error('Error fetching files:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleStartMLWorkflow = (file: UploadedFile) => {
    navigate(`/dataset/${file.id}/config`);
  };

  const handleFileSelect = (file: UploadedFile) => {
    navigate(`/files/${file.id}`);
  };

  const mlWorkflowSteps = [
    {
      title: '1. Upload Dataset',
      description: 'Upload CSV or Excel files with your data',
      icon: <Upload className="h-6 w-6" />,
      color: 'bg-blue-100 text-blue-600',
      action: () => navigate('/upload')
    },
    {
      title: '2. Configure Dataset',
      description: 'Select target variable and problem type',
      icon: <Target className="h-6 w-6" />,
      color: 'bg-purple-100 text-purple-600'
    },
    {
      title: '3. Choose Algorithms',
      description: 'Select ML algorithms and hyperparameters',
      icon: <Settings className="h-6 w-6" />,
      color: 'bg-green-100 text-green-600'
    },
    {
      title: '4. Train & Compare',
      description: 'Train models and compare results',
      icon: <BarChart3 className="h-6 w-6" />,
      color: 'bg-yellow-100 text-yellow-600'
    }
  ];

  const algorithmCategories = [
    {
      title: 'Classification',
      description: 'Predict categories or classes',
      algorithms: ['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM', 'KNN'],
      color: 'bg-blue-50 border-blue-200',
      icon: <Target className="h-5 w-5 text-blue-600" />
    },
    {
      title: 'Regression',
      description: 'Predict continuous values',
      algorithms: ['Linear Regression', 'Random Forest', 'Decision Tree', 'SVR', 'KNN'],
      color: 'bg-green-50 border-green-200',
      icon: <TrendingUp className="h-5 w-5 text-green-600" />
    }
  ];

  if (isLoading) {
    return (
      <div className="px-4 py-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading datasets...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-6">
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-3xl font-bold text-gray-900 mb-3 flex items-center justify-center">
              <Brain className="h-8 w-8 text-purple-600 mr-3" />
              ML Training Center
            </h1>
            <p className="text-lg text-gray-600 mb-8">
              Train machine learning models on your datasets with automated preprocessing and algorithm comparison
            </p>
          </motion.div>
        </div>

        {/* Quick Actions */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            onClick={() => navigate('/upload')}
            className="flex items-center"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Dataset
          </Button>
          <Button
            variant="outline"
            onClick={fetchFiles}
            className="flex items-center"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Datasets
          </Button>
          <Button
            variant="outline"
            onClick={() => navigate('/results')}
            className="flex items-center"
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            View Results
          </Button>
        </div>

        {/* Workflow Steps */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Zap className="h-5 w-5 mr-2 text-purple-600" />
              ML Training Workflow
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {mlWorkflowSteps.map((step, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="text-center"
                >
                  <div className={`w-12 h-12 rounded-full ${step.color} flex items-center justify-center mx-auto mb-3`}>
                    {step.icon}
                  </div>
                  <h3 className="font-medium text-gray-900 mb-2">{step.title}</h3>
                  <p className="text-sm text-gray-600 mb-3">{step.description}</p>
                  {step.action && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={step.action}
                      className="text-xs"
                    >
                      Get Started
                    </Button>
                  )}
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Available Datasets */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center">
                <Database className="h-5 w-5 mr-2 text-green-600" />
                Available Datasets ({files.length})
              </CardTitle>
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate('/files?filter=datasets')}
                className="text-sm"
              >
                <FileText className="h-4 w-4 mr-1" />
                View All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {error ? (
              <div className="flex items-center text-red-500 p-4">
                <AlertCircle className="h-5 w-5 mr-2" />
                <span>{error}</span>
              </div>
            ) : files.length === 0 ? (
              <div className="text-center py-8">
                <Database className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Datasets Found</h3>
                <p className="text-gray-600 mb-4">
                  Upload CSV or Excel files to start training ML models.
                </p>
                <Button onClick={() => navigate('/upload')}>
                  <Upload className="h-4 w-4 mr-1" />
                  Upload Dataset
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {files.slice(0, 6).map((file, index) => (
                  <motion.div
                    key={file.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="relative"
                  >
                    <FileCard
                      file={file}
                      onClick={handleFileSelect}
                    />
                    <div className="mt-2 flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleFileSelect(file);
                        }}
                        className="flex-1 text-xs"
                      >
                        <FileText className="h-3 w-3 mr-1" />
                        View
                      </Button>
                      <Button
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartMLWorkflow(file);
                        }}
                        className="flex-1 text-xs"
                      >
                        <Play className="h-3 w-3 mr-1" />
                        Train
                      </Button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Supported Algorithms */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="h-5 w-5 mr-2 text-blue-600" />
              Supported Algorithms
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {algorithmCategories.map((category, index) => (
                <motion.div
                  key={category.title}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className={`p-6 rounded-lg border-2 ${category.color}`}
                >
                  <div className="flex items-center mb-3">
                    {category.icon}
                    <h3 className="text-lg font-medium text-gray-900 ml-2">{category.title}</h3>
                  </div>
                  <p className="text-gray-600 mb-4">{category.description}</p>
                  <div className="space-y-2">
                    {category.algorithms.map((algorithm) => (
                      <div key={algorithm} className="text-sm text-gray-700 flex items-center">
                        <div className="w-2 h-2 bg-gray-400 rounded-full mr-2"></div>
                        {algorithm}
                      </div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Getting Started Card */}
        <Card className="bg-gradient-to-r from-purple-50 to-blue-50 border-purple-200">
          <CardContent className="p-6">
            <div className="text-center">
              <Brain className="h-12 w-12 text-purple-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to Start?</h3>
              <p className="text-gray-600 mb-4">
                Upload a dataset or select from your existing files to begin training machine learning models.
              </p>
              <div className="flex justify-center space-x-3">
                <Button onClick={() => navigate('/upload')}>
                  <Upload className="h-4 w-4 mr-1" />
                  Upload Dataset
                </Button>
                <Button variant="outline" onClick={() => navigate('/files')}>
                  <Database className="h-4 w-4 mr-1" />
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

export default MLTrainingPage; 