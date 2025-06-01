import React from 'react';
import { LineChart, BarChart, PieChart, AlertCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { DatasetProfileSummary } from '../../types';
import { formatBytes, getPercentage } from '../../utils/helpers';
import Spinner from '../ui/Spinner';

interface DataProfileSummaryProps {
  profile: DatasetProfileSummary | null;
  isLoading: boolean;
  error: string | null;
  className?: string;
}

const DataProfileSummary = ({
  profile,
  isLoading,
  error,
  className
}: DataProfileSummaryProps) => {
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Dataset Profile</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center p-8">
          <Spinner size="md" />
          <span className="ml-3 text-gray-500">Analyzing dataset...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Dataset Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center text-red-500 p-4">
            <AlertCircle className="h-5 w-5 mr-2" />
            <span>{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!profile) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Dataset Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center text-gray-500 p-8">
            No profile data available
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate missing data percentage
  const missingDataPercent = getPercentage(
    profile.missing_cells,
    profile.total_rows * profile.total_columns
  );

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Dataset Profile</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="bg-blue-100 rounded-full p-2">
                <BarChart className="h-5 w-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Rows</p>
                <p className="text-xl font-semibold text-gray-900">{profile.total_rows}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="bg-purple-100 rounded-full p-2">
                <LineChart className="h-5 w-5 text-purple-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Columns</p>
                <p className="text-xl font-semibold text-gray-900">{profile.total_columns}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-amber-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="bg-amber-100 rounded-full p-2">
                <PieChart className="h-5 w-5 text-amber-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Missing Data</p>
                <p className="text-xl font-semibold text-gray-900">{missingDataPercent}%</p>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="bg-green-100 rounded-full p-2">
                <LineChart className="h-5 w-5 text-green-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Size</p>
                <p className="text-xl font-semibold text-gray-900">{formatBytes(parseInt(profile.memory_usage))}</p>
              </div>
            </div>
          </div>
        </div>
        
        <h3 className="font-medium text-gray-900 mb-3">Column Information</h3>
        {profile.columns && profile.columns.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Column Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Data Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Missing
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Unique Values
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Sample
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {profile.columns?.map((column, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {column.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                        {column.data_type}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {column.missing_count} ({getPercentage(column.missing_count, profile.total_rows)}%)
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {column.unique_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 max-w-xs truncate">
                      {column.sample_values?.slice(0, 3).join(', ') || 'No samples'}
                      {column.sample_values && column.sample_values.length > 3 && '...'}
                    </td>
                  </tr>
                )) || []}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex items-center justify-center text-gray-500 p-8 border border-gray-200 rounded-lg">
            <AlertCircle className="h-5 w-5 mr-2" />
            No column information available
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DataProfileSummary;
