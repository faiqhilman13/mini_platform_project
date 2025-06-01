import React, { useState } from 'react';
import { 
  Download,
  Save,
  FileText,
  Database,
  Code,
  Settings,
  CheckCircle,
  AlertCircle,
  Loader2,
  Filter,
  Calendar,
  User,
  Tag
} from 'lucide-react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { MLPipelineRun, MLModel } from '../../types';

interface ResultsExportProps {
  pipelineRun: MLPipelineRun;
  models: MLModel[];
  className?: string;
}

interface ExportFormat {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  fileExtension: string;
  mimeType: string;
}

const exportFormats: ExportFormat[] = [
  {
    id: 'json',
    name: 'JSON Report',
    description: 'Complete results in JSON format with all metrics and configurations',
    icon: <Code className="h-4 w-4" />,
    fileExtension: 'json',
    mimeType: 'application/json'
  },
  {
    id: 'csv',
    name: 'CSV Summary',
    description: 'Model comparison table in CSV format for spreadsheet analysis',
    icon: <Database className="h-4 w-4" />,
    fileExtension: 'csv',
    mimeType: 'text/csv'
  },
  {
    id: 'config',
    name: 'Configuration',
    description: 'Pipeline configuration for reproducing this experiment',
    icon: <Settings className="h-4 w-4" />,
    fileExtension: 'json',
    mimeType: 'application/json'
  }
];

const ResultsExport = ({ pipelineRun, models, className }: ResultsExportProps) => {
  const [selectedFormats, setSelectedFormats] = useState<Set<string>>(new Set(['json']));
  const [exportStatus, setExportStatus] = useState<'idle' | 'exporting' | 'success' | 'error'>('idle');

  const toggleFormat = (formatId: string) => {
    const newSelected = new Set(selectedFormats);
    if (newSelected.has(formatId)) {
      newSelected.delete(formatId);
    } else {
      newSelected.add(formatId);
    }
    setSelectedFormats(newSelected);
  };

  const generateJSONReport = () => {
    const bestModel = models.find(m => m.model_id === pipelineRun.best_model_id);
    
    return {
      export_info: {
        generated_at: new Date().toISOString(),
        pipeline_run_id: pipelineRun.run_uuid,
        export_version: '1.0'
      },
      experiment_summary: {
        problem_type: pipelineRun.problem_type,
        target_variable: pipelineRun.target_variable,
        created_at: pipelineRun.created_at,
        status: pipelineRun.status,
        total_models: models.length,
        best_model_id: pipelineRun.best_model_id,
        best_algorithm: bestModel?.algorithm_name || null
      },
      preprocessing_config: pipelineRun.preprocessing_config,
      algorithms_config: pipelineRun.algorithms_config,
      models: models.map(model => ({
        model_id: model.model_id,
        algorithm_name: model.algorithm_name,
        hyperparameters: model.hyperparameters,
        performance_metrics: model.performance_metrics,
        training_time: model.training_time,
        feature_importance: model.feature_importance || null
      })),
      pipeline_metrics: pipelineRun.metrics || {}
    };
  };

  const generateCSVSummary = () => {
    const headers = [
      'Algorithm',
      'Model ID',
      'Training Time (s)',
      ...(pipelineRun.problem_type === 'CLASSIFICATION' 
        ? ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        : ['R² Score', 'MAE', 'RMSE', 'MSE']
      ),
      'Is Best'
    ];

    const rows = models.map(model => {
      const metrics = model.performance_metrics;
      const isBest = model.model_id === pipelineRun.best_model_id;
      
      return [
        model.algorithm_name,
        model.model_id,
        model.training_time?.toFixed(4) || 'N/A',
        ...(pipelineRun.problem_type === 'CLASSIFICATION'
          ? [
              metrics.accuracy?.toFixed(4) || 'N/A',
              metrics.precision?.toFixed(4) || 'N/A',
              metrics.recall?.toFixed(4) || 'N/A',
              metrics.f1_score?.toFixed(4) || 'N/A',
              metrics.roc_auc?.toFixed(4) || 'N/A'
            ]
          : [
              metrics.r2?.toFixed(4) || 'N/A',
              metrics.mae?.toFixed(4) || 'N/A',
              metrics.rmse?.toFixed(4) || 'N/A',
              metrics.mse?.toFixed(4) || 'N/A'
            ]
        ),
        isBest ? 'Yes' : 'No'
      ];
    });

    return [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');
  };

  const generateConfigurationExport = () => {
    return {
      pipeline_configuration: {
        problem_type: pipelineRun.problem_type,
        target_variable: pipelineRun.target_variable,
        preprocessing_config: pipelineRun.preprocessing_config,
        algorithms_config: pipelineRun.algorithms_config
      },
      reproduction_instructions: {
        description: 'Use this configuration to reproduce the same ML pipeline',
        steps: [
          '1. Upload your dataset',
          '2. Apply the preprocessing configuration',
          '3. Configure algorithms with the provided hyperparameters',
          '4. Run the ML pipeline'
        ]
      },
      exported_at: new Date().toISOString()
    };
  };

  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExport = async () => {
    if (selectedFormats.size === 0) return;

    setExportStatus('exporting');
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate export time

      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
      const baseFilename = `ml_results_${pipelineRun.run_uuid.slice(0, 8)}_${timestamp}`;

      for (const formatId of selectedFormats) {
        const format = exportFormats.find(f => f.id === formatId);
        if (!format) continue;

        let content = '';
        let filename = '';

        switch (formatId) {
          case 'json':
            content = JSON.stringify(generateJSONReport(), null, 2);
            filename = `${baseFilename}.${format.fileExtension}`;
            break;
          case 'csv':
            content = generateCSVSummary();
            filename = `${baseFilename}_summary.${format.fileExtension}`;
            break;
          case 'config':
            content = JSON.stringify(generateConfigurationExport(), null, 2);
            filename = `${baseFilename}_config.${format.fileExtension}`;
            break;
        }

        downloadFile(content, filename, format.mimeType);
      }

      setExportStatus('success');
      setTimeout(() => setExportStatus('idle'), 3000);
    } catch (error) {
      console.error('Export failed:', error);
      setExportStatus('error');
      setTimeout(() => setExportStatus('idle'), 3000);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'exporting':
        return <Loader2 className="h-4 w-4 animate-spin" />;
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-600" />;
      default:
        return null;
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Download className="h-5 w-5 mr-2 text-blue-600" />
          Export Results
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Format Selection */}
          <div>
            <h3 className="text-sm font-medium text-gray-900 mb-3">Select Export Formats</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {exportFormats.map((format) => (
                <motion.button
                  key={format.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => toggleFormat(format.id)}
                  className={cn(
                    'p-4 text-left border rounded-lg transition-all',
                    selectedFormats.has(format.id)
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  )}
                >
                  <div className="flex items-start space-x-3">
                    <div className={cn(
                      'p-2 rounded-md',
                      selectedFormats.has(format.id) ? 'bg-blue-100' : 'bg-gray-100'
                    )}>
                      {format.icon}
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900">{format.name}</h4>
                      <p className="text-sm text-gray-300 mt-1">{format.description}</p>
                    </div>
                    {selectedFormats.has(format.id) && (
                      <CheckCircle className="h-5 w-5 text-blue-600 flex-shrink-0" />
                    )}
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Export Actions */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-200">
            <div className="text-sm text-gray-300">
              {selectedFormats.size} format{selectedFormats.size !== 1 ? 's' : ''} selected
            </div>
            
            <Button
              onClick={handleExport}
              disabled={selectedFormats.size === 0 || exportStatus === 'exporting'}
              className="flex items-center"
            >
              {getStatusIcon(exportStatus)}
              <span className="ml-2">
                {exportStatus === 'exporting' ? 'Exporting...' :
                 exportStatus === 'success' ? 'Exported!' :
                 exportStatus === 'error' ? 'Export Failed' : 'Export Results'}
              </span>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResultsExport; 
