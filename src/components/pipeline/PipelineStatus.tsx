import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, Clock, Loader } from 'lucide-react';
import { PipelineRun } from '../../types';
import { PIPELINE_STATUS_COLORS } from '../../utils/constants';
import { cn, formatDate } from '../../utils/helpers';
import { Card, CardContent } from '../ui/Card';

interface PipelineStatusProps {
  pipelineRun: PipelineRun;
  refreshInterval?: number; // in milliseconds
  onRefresh?: () => void;
  onClick?: (pipelineRun: PipelineRun) => void;
  className?: string;
}

const PipelineStatus = ({  pipelineRun,  refreshInterval = 5000,  onRefresh,  onClick,  className}: PipelineStatusProps) => {
  const [progress, setProgress] = useState(0);

  // Early return if pipelineRun is not properly defined
  if (!pipelineRun || !pipelineRun.run_uuid) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-center h-16">
            <Loader className="h-6 w-6 text-gray-400 animate-spin" />
            <span className="ml-2 text-sm text-gray-500">Loading pipeline...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Set up auto-refresh for in-progress pipelines
  useEffect(() => {
    if (pipelineRun.status === 'PROCESSING' || pipelineRun.status === 'QUEUED') {
      const interval = setInterval(() => {
        if (onRefresh) {
          onRefresh();
        }
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [pipelineRun.status, refreshInterval, onRefresh]);

  // Simulate progress for PROCESSING status
  useEffect(() => {
    if (pipelineRun.status === 'PROCESSING') {
      // Reset progress when status changes to PROCESSING
      setProgress(0);
      
      // Animate progress from 0 to 90% (not 100% since we don't know when it will complete)
      const duration = 30000; // 30 seconds to reach 90%
      const interval = 100; // update every 100ms
      const step = (90 / (duration / interval));
      
      const timer = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + step;
          return newProgress >= 90 ? 90 : newProgress;
        });
      }, interval);
      
      return () => clearInterval(timer);
    } else if (pipelineRun.status === 'COMPLETED') {
      setProgress(100);
    } else if (pipelineRun.status === 'FAILED') {
      // Set partial progress for failed jobs
      setProgress(prev => prev > 0 ? prev : 30);
    } else if (pipelineRun.status === 'QUEUED') {
      setProgress(5);
    }
  }, [pipelineRun.status]);

  const renderStatusIcon = () => {
    switch (pipelineRun.status) {
      case 'COMPLETED':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      case 'FAILED':
        return <XCircle className="h-6 w-6 text-red-500" />;
      case 'QUEUED':
        return <Clock className="h-6 w-6 text-amber-500" />;
      case 'PROCESSING':
        return (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            <Loader className="h-6 w-6 text-blue-500" />
          </motion.div>
        );
      default:
        return null;
    }
  };

    const handleCardClick = () => {    if (pipelineRun.status === 'COMPLETED' && onClick) {      onClick(pipelineRun);    }  };  return (    <div      className={cn(        pipelineRun.status === 'COMPLETED' && onClick && 'cursor-pointer hover:shadow-md transition-shadow'      )}      onClick={handleCardClick}    >      <Card className={className}>        <CardContent className="p-4">
        <div className="flex items-center mb-4">
          {renderStatusIcon()}
          <div className="ml-3">
            <div className="flex items-center">
              <span className="text-sm font-medium text-gray-900">
                {pipelineRun.pipeline_type ? pipelineRun.pipeline_type.replace('_', ' ') : 'Unknown Pipeline'}
              </span>
              <span
                className={cn(
                  'ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                  PIPELINE_STATUS_COLORS[pipelineRun.status] || 'bg-gray-100 text-gray-800'
                )}
              >
                {pipelineRun.status || 'UNKNOWN'}
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Started: {formatDate(pipelineRun.created_at)}
            </p>
          </div>
        </div>

        <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
          <motion.div
            className={cn(
              "h-2.5 rounded-full",
              pipelineRun.status === 'COMPLETED' ? 'bg-green-600' :
              pipelineRun.status === 'FAILED' ? 'bg-red-600' :
              'bg-blue-600'
            )}
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          ></motion.div>
        </div>

        <div className="flex justify-between text-xs text-gray-500">
          <span>Run ID: {pipelineRun.run_uuid.substring(0, 8)}...</span>
          <span>
            {pipelineRun.status === 'COMPLETED' && 'Completed'}
            {pipelineRun.status === 'PROCESSING' && 'Processing...'}
            {pipelineRun.status === 'QUEUED' && 'Queued'}
            {pipelineRun.status === 'FAILED' && 'Failed'}
          </span>
        </div>

        {pipelineRun.status === 'FAILED' && pipelineRun.error_message && (
          <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-xs">
            <p className="font-medium">Error:</p>
            <p className="mt-1">{pipelineRun.error_message}</p>
          </div>
                )}      </CardContent>    </Card>    </div>  );};

export default PipelineStatus;