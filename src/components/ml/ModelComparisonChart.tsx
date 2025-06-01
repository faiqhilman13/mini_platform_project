import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { MLModel } from '../../types';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { getRandomColor } from '../../utils/helpers';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface ModelComparisonChartProps {
  models: MLModel[];
  bestModelId?: string;
  className?: string;
}

const ModelComparisonChart = ({
  models,
  bestModelId,
  className
}: ModelComparisonChartProps) => {
  // Return null if no models
  if (!models || models.length === 0) {
    return null;
  }

  // Get display names for algorithms
  const getDisplayName = (name: string): string => {
    return name.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  // Get all metrics from the first model
  const metrics = Object.keys(models[0].performance_metrics);

  // Create datasets for each metric
  const datasets = metrics.map((metric, index) => {
    return {
      label: metric.charAt(0).toUpperCase() + metric.slice(1),
      data: models.map(model => model.performance_metrics[metric]),
      backgroundColor: getRandomColor(index),
      borderWidth: 1,
    };
  });

  // Chart data
  const data = {
    labels: models.map(model => getDisplayName(model.algorithm_name)),
    datasets,
  };

  // Chart options
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Model Performance Metrics Comparison',
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value.toFixed(4)}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value: any) {
            return value.toFixed(2);
          }
        }
      }
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Model Performance Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <Bar data={data} options={options} />
        <div className="mt-4 text-xs text-center text-gray-500">
          Higher values generally indicate better performance (except for error metrics like MSE)
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelComparisonChart;
