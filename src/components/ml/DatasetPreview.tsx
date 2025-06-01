import React, { useState, useMemo } from 'react';
import { 
  Table2, 
  AlertCircle, 
  Info, 
  Hash, 
  Type, 
  Calendar,
  ToggleLeft,
  ToggleRight,
  Eye,
  EyeOff,
  BarChart3
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import Spinner from '../ui/Spinner';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { DatasetColumn } from '../../types';

interface DatasetPreviewProps {
  data: any[] | null;
  columns?: DatasetColumn[]; // Column metadata for enhanced display
  isLoading: boolean;
  error: string | null;
  maxRows?: number;
  maxCols?: number;
  className?: string;
  showStatistics?: boolean;
}

// Column type detection for when metadata is not provided
const detectColumnType = (values: any[]): { type: string; icon: React.ReactNode; color: string } => {
  const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
  
  if (nonNullValues.length === 0) {
    return { type: 'unknown', icon: <Type className="h-3 w-3" />, color: 'text-gray-400' };
  }

  // Check if all values are numbers
  const numericValues = nonNullValues.filter(v => !isNaN(Number(v)));
  if (numericValues.length === nonNullValues.length) {
    // Check if integers vs floats
    const hasDecimals = numericValues.some(v => Number(v) % 1 !== 0);
    return { 
      type: hasDecimals ? 'float' : 'integer', 
      icon: <Hash className="h-3 w-3" />, 
      color: 'text-blue-500' 
    };
  }

  // Check if values look like dates
  const dateValues = nonNullValues.filter(v => !isNaN(Date.parse(String(v))));
  if (dateValues.length / nonNullValues.length > 0.8) {
    return { type: 'date', icon: <Calendar className="h-3 w-3" />, color: 'text-purple-500' };
  }

  // Check if boolean-like values
  const booleanValues = nonNullValues.filter(v => 
    ['true', 'false', '1', '0', 'yes', 'no'].includes(String(v).toLowerCase())
  );
  if (booleanValues.length === nonNullValues.length) {
    return { type: 'boolean', icon: <ToggleLeft className="h-3 w-3" />, color: 'text-green-500' };
  }

  // Default to text
  return { type: 'text', icon: <Type className="h-3 w-3" />, color: 'text-gray-300' };
};

// Calculate basic statistics for a column
const calculateColumnStats = (values: any[], columnType: string) => {
  const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
  const nullCount = values.length - nonNullValues.length;
  const nullPercentage = (nullCount / values.length) * 100;

  const stats: any = {
    total: values.length,
    missing: nullCount,
    missingPercentage: nullPercentage,
    unique: new Set(nonNullValues).size
  };

  if (columnType === 'integer' || columnType === 'float') {
    const numericValues = nonNullValues.map(Number).filter(v => !isNaN(v));
    if (numericValues.length > 0) {
      stats.min = Math.min(...numericValues);
      stats.max = Math.max(...numericValues);
      stats.mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
      
      // Calculate median
      const sorted = [...numericValues].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      stats.median = sorted.length % 2 === 0 
        ? (sorted[mid - 1] + sorted[mid]) / 2 
        : sorted[mid];
    }
  }

  return stats;
};

const DatasetPreview = ({
  data,
  columns,
  isLoading,
  error,
  maxRows = 10,
  maxCols = 8,
  className,
  showStatistics = true
}: DatasetPreviewProps) => {
  const [showStats, setShowStats] = useState(false);
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(new Set());

  // Memoized column analysis
  const columnAnalysis = useMemo(() => {
    if (!data || data.length === 0) return {};
    
    const analysis: Record<string, any> = {};
    const columnNames = Object.keys(data[0]);
    
    columnNames.forEach(colName => {
      const values = data.map(row => row[colName]);
      
      // Use provided metadata or detect type
      let columnInfo;
      if (columns) {
        const metadata = columns.find(col => col.name === colName);
        if (metadata) {
          columnInfo = {
            type: metadata.data_type,
            icon: <Hash className="h-3 w-3" />,
            color: metadata.data_type.includes('int') || metadata.data_type.includes('float') 
              ? 'text-blue-500' 
              : 'text-gray-300'
          };
        }
      }
      
      if (!columnInfo) {
        columnInfo = detectColumnType(values);
      }
      
      analysis[colName] = {
        ...columnInfo,
        stats: calculateColumnStats(values, columnInfo.type)
      };
    });
    
    return analysis;
  }, [data, columns]);

  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Table2 className="h-5 w-5 mr-2 text-blue-600" />
            Dataset Preview
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center p-8">
          <Spinner size="md" />
          <span className="ml-3 text-gray-500">Loading dataset preview...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Table2 className="h-5 w-5 mr-2 text-blue-600" />
            Dataset Preview
          </CardTitle>
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

  if (!data || data.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Table2 className="h-5 w-5 mr-2 text-blue-600" />
            Dataset Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center text-gray-500 p-8">
            No data available
          </div>
        </CardContent>
      </Card>
    );
  }

  // Get column headers from the first row
  const allColumns = Object.keys(data[0]);
  
  // Limit the number of columns to display
  const displayColumns = allColumns.slice(0, maxCols);
  const hasMoreColumns = allColumns.length > maxCols;

  // Limit the number of rows to display
  const displayRows = data.slice(0, maxRows);
  const hasMoreRows = data.length > maxRows;

  const toggleColumnVisibility = (column: string) => {
    const newVisible = new Set(visibleColumns);
    if (newVisible.has(column)) {
      newVisible.delete(column);
    } else {
      newVisible.add(column);
    }
    setVisibleColumns(newVisible);
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-0">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Table2 className="h-5 w-5 mr-2 text-blue-600" />
            Dataset Preview
          </CardTitle>
          {showStatistics && (
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowStats(!showStats)}
                className="text-xs"
              >
                <BarChart3 className="h-3 w-3 mr-1" />
                {showStats ? 'Hide Stats' : 'Show Stats'}
              </Button>
            </div>
          )}
        </div>
        
        {/* Dataset Summary */}
        <div className="flex items-center space-x-4 text-sm text-gray-300 mt-2">
          <span className="flex items-center">
            <Info className="h-4 w-4 mr-1" />
            {data.length} rows × {allColumns.length} columns
          </span>
          {hasMoreRows && (
            <span className="text-amber-600">
              Showing first {maxRows} rows
            </span>
          )}
          {hasMoreColumns && (
            <span className="text-amber-600">
              Showing first {maxCols} columns
            </span>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Statistics Panel */}
        {showStats && (
          <div className="mb-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Column Statistics</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {displayColumns.map((column) => {
                const analysis = columnAnalysis[column];
                if (!analysis) return null;
                
                const stats = analysis.stats;
                return (
                  <div key={column} className="bg-white p-3 rounded border">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className={analysis.color}>
                          {analysis.icon}
                        </span>
                        <span className="font-medium text-sm text-gray-900 truncate">
                          {column}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500 capitalize">
                        {analysis.type}
                      </span>
                    </div>
                    
                    <div className="space-y-1 text-xs text-gray-300">
                      <div className="flex justify-between">
                        <span>Unique:</span>
                        <span>{stats.unique}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Missing:</span>
                        <span className={stats.missing > 0 ? 'text-red-500' : 'text-green-500'}>
                          {stats.missing} ({stats.missingPercentage.toFixed(1)}%)
                        </span>
                      </div>
                      
                      {(analysis.type === 'integer' || analysis.type === 'float') && stats.min !== undefined && (
                        <>
                          <div className="flex justify-between">
                            <span>Range:</span>
                            <span>{stats.min.toFixed(2)} - {stats.max.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Mean:</span>
                            <span>{stats.mean.toFixed(2)}</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Data Table */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {displayColumns.map((column, idx) => {
                  const analysis = columnAnalysis[column];
                  const stats = analysis?.stats;
                  
                  return (
                    <th
                      key={idx}
                      className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      <div className="flex flex-col">
                        <div className="flex items-center space-x-2">
                          {analysis && (
                            <span className={analysis.color}>
                              {analysis.icon}
                            </span>
                          )}
                          <span className="truncate">{column}</span>
                        </div>
                        
                        {/* Missing value indicator */}
                        {stats && stats.missing > 0 && (
                          <div className="mt-1">
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-700">
                              {stats.missing} missing
                            </span>
                          </div>
                        )}
                        
                        {/* Data type indicator */}
                        {analysis && (
                          <div className="mt-1">
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-300 capitalize">
                              {analysis.type}
                            </span>
                          </div>
                        )}
                      </div>
                    </th>
                  );
                })}
                {hasMoreColumns && (
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500">
                    +{allColumns.length - maxCols} more columns
                  </th>
                )}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {displayRows.map((row, rowIdx) => (
                <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  {displayColumns.map((column, colIdx) => {
                    const value = row[column];
                    const analysis = columnAnalysis[column];
                    const isNull = value === null || value === undefined || value === '';
                    
                    return (
                      <td
                        key={colIdx}
                        className={cn(
                          'px-4 py-3 whitespace-nowrap text-sm',
                          analysis?.type === 'integer' || analysis?.type === 'float' 
                            ? 'text-right' 
                            : 'text-left',
                          isNull ? 'text-gray-400 italic' : 'text-gray-900'
                        )}
                      >
                        {isNull ? (
                          <span className="bg-red-50 text-red-600 px-2 py-1 rounded text-xs">
                            null
                          </span>
                        ) : (
                          <span className={cn(
                            analysis?.type === 'float' && 'font-mono',
                            analysis?.type === 'integer' && 'font-mono'
                          )}>
                            {analysis?.type === 'float' 
                              ? Number(value).toFixed(2)
                              : String(value)
                            }
                          </span>
                        )}
                      </td>
                    );
                  })}
                  {hasMoreColumns && (
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-400">
                      ...
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          
          {hasMoreRows && (
            <div className="py-3 px-4 text-sm text-gray-500 border-t bg-gray-50">
              <div className="flex items-center justify-between">
                <span>Showing {maxRows} of {data.length} rows</span>
                <span className="text-xs text-gray-400">
                  Scroll horizontally to see more columns
                </span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default DatasetPreview;
