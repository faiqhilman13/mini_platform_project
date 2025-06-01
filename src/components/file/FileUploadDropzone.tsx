import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  FileText, 
  Table, 
  Table2, 
  File,
  Braces,
  AlertCircle, 
  CheckCircle2,
  X,
  Eye
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';
import { FileType } from '../../types';

interface FileUploadDropzoneProps {
  onUpload: (file: File) => Promise<void>;
  acceptedFileTypes?: string[];
  maxFileSize?: number; // in bytes
  className?: string;
  showPreview?: boolean;
}

interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

const FileUploadDropzone = ({
  onUpload,
  acceptedFileTypes = ['.pdf', '.csv', '.xlsx', '.txt', '.json'],
  maxFileSize = 100 * 1024 * 1024, // 100MB default (increased for datasets)
  className,
  showPreview = true,
}: FileUploadDropzoneProps) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [showFilePreview, setShowFilePreview] = useState(false);

  // Get file type icon
  const getFileTypeIcon = (filename: string, size: number = 12) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    const iconProps = { className: `h-${size/4} w-${size/4}` };
    
    switch (extension) {
      case 'pdf':
        return <FileText {...iconProps} className={`${iconProps.className} text-red-500`} />;
      case 'csv':
        return <Table {...iconProps} className={`${iconProps.className} text-green-500`} />;
      case 'xlsx':
      case 'xls':
        return <Table2 {...iconProps} className={`${iconProps.className} text-green-600`} />;
      case 'json':
        return <Braces {...iconProps} className={`${iconProps.className} text-blue-500`} />;
      case 'txt':
        return <FileText {...iconProps} className={`${iconProps.className} text-gray-500`} />;
      default:
        return <File {...iconProps} className={`${iconProps.className} text-gray-400`} />;
    }
  };

  // Get file type label and description
  const getFileTypeInfo = (filename: string): { type: FileType; label: string; description: string } => {
    const extension = filename.split('.').pop()?.toLowerCase();
    
    switch (extension) {
      case 'pdf':
        return { 
          type: 'pdf', 
          label: 'PDF Document', 
          description: 'For document analysis, RAG chatbot, summarization' 
        };
      case 'csv':
        return { 
          type: 'csv', 
          label: 'CSV Dataset', 
          description: 'For machine learning training and data analysis' 
        };
      case 'xlsx':
      case 'xls':
        return { 
          type: 'xlsx', 
          label: 'Excel Spreadsheet', 
          description: 'For machine learning training and data analysis' 
        };
      case 'json':
        return { 
          type: 'json', 
          label: 'JSON File', 
          description: 'For structured data processing' 
        };
      case 'txt':
        return { 
          type: 'text', 
          label: 'Text File', 
          description: 'For text analysis and classification' 
        };
      default:
        return { 
          type: 'unknown', 
          label: 'Unknown File', 
          description: 'File type not recognized' 
        };
    }
  };

  // Validate file
  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize) {
      return `File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds the maximum limit of ${(maxFileSize / (1024 * 1024)).toFixed(0)}MB`;
    }

    // Check file type
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedFileTypes.includes(extension)) {
      return `File type ${extension} is not supported. Supported types: ${acceptedFileTypes.join(', ')}`;
    }

    // Additional validation for datasets
    if (['.csv', '.xlsx', '.xls'].includes(extension)) {
      if (file.size < 100) { // Very small files are likely empty
        return 'Dataset file appears to be empty or corrupted';
      }
    }

    return null;
  };

  // Generate preview data for CSV files
  const generatePreview = async (file: File) => {
    if (!showPreview) return;
    
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (extension === '.csv') {
      try {
        const text = await file.text();
        const lines = text.split('\n').slice(0, 5); // First 5 lines
        const preview = lines.map(line => line.split(',').slice(0, 5)); // First 5 columns
        setPreviewData(preview);
      } catch (error) {
        console.warn('Failed to generate preview:', error);
      }
    }
  };

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      const file = acceptedFiles[0];
      setSelectedFile(file);
      
      // Validate file
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }

      // Generate preview
      await generatePreview(file);

      // Clear previous errors
      setError(null);

      try {
        setIsUploading(true);
        setUploadProgress({ loaded: 0, total: file.size, percentage: 0 });
        
        // Simulate upload progress
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (!prev) return null;
            const newLoaded = Math.min(prev.loaded + file.size / 20, file.size);
            const newPercentage = Math.round((newLoaded / file.size) * 100);
            return {
              loaded: newLoaded,
              total: file.size,
              percentage: newPercentage
            };
          });
        }, 100);

        await onUpload(file);
        
        clearInterval(progressInterval);
        setUploadProgress({ loaded: file.size, total: file.size, percentage: 100 });
        
        // Show success briefly before clearing
        setTimeout(() => {
          setUploadProgress(null);
          setSelectedFile(null);
          setPreviewData(null);
        }, 1500);
        
      } catch (error) {
        setError(error instanceof Error ? error.message : 'An error occurred during upload');
        setUploadProgress(null);
      } finally {
        setIsUploading(false);
      }
    },
    [onUpload, maxFileSize, acceptedFileTypes, showPreview]
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
  } = useDropzone({
    onDrop,
    accept: acceptedFileTypes.reduce((acc, type) => {
      if (type === '.pdf') acc['application/pdf'] = [];
      if (type === '.csv') acc['text/csv'] = [];
      if (type === '.xlsx') acc['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] = [];
      if (type === '.xls') acc['application/vnd.ms-excel'] = [];
      if (type === '.txt') acc['text/plain'] = [];
      if (type === '.json') acc['application/json'] = [];
      return acc;
    }, {} as Record<string, string[]>),
    maxFiles: 1,
  });

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewData(null);
    setError(null);
    setUploadProgress(null);
  };

  return (
    <div className={className}>
      <div
        {...getRootProps()}
        className={cn(
          'flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-8 transition-all duration-200',
          'cursor-pointer relative overflow-hidden',
          isDragActive ? 'border-blue-500 bg-blue-50 scale-105' : 'border-gray-300 bg-gray-50',
          isDragAccept ? 'border-green-500 bg-green-50' : '',
          isDragReject ? 'border-red-500 bg-red-50' : '',
          isUploading ? 'opacity-75 pointer-events-none' : '',
          selectedFile && !isUploading ? 'border-green-500 bg-green-50' : ''
        )}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {uploadProgress && uploadProgress.percentage === 100 ? (
            <motion.div
              key="success"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              className="text-center"
            >
              <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <p className="text-lg font-medium text-green-700">Upload Complete!</p>
              <p className="text-sm text-green-600">File uploaded successfully</p>
            </motion.div>
          ) : (
            <motion.div
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center w-full"
            >
              <motion.div 
                initial={{ scale: 1 }}
                animate={{ 
                  scale: isDragActive ? 1.1 : 1,
                  y: isDragActive ? -10 : 0
                }}
                transition={{ type: 'spring', stiffness: 300, damping: 15 }}
                className="mb-4"
              >
                {isDragReject ? (
                  <AlertCircle className="h-12 w-12 text-red-500 mx-auto" />
                ) : isDragAccept ? (
                  <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto" />
                ) : (
                  <Upload className="h-12 w-12 text-blue-500 mx-auto" />
                )}
              </motion.div>

              {selectedFile && !uploadProgress ? (
                <div className="mb-4 p-3 bg-white rounded-lg border border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getFileTypeIcon(selectedFile.name, 16)}
                      <div className="text-left">
                        <p className="font-medium text-gray-900 text-sm">{selectedFile.name}</p>
                        <p className="text-xs text-gray-500">
                          {getFileTypeInfo(selectedFile.name).label} • {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        clearSelection();
                      }}
                      className="p-1 hover:bg-gray-100 rounded-full"
                    >
                      <X className="h-4 w-4 text-gray-400" />
                    </button>
                  </div>
                  
                  {previewData && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-xs font-medium text-gray-300">Data Preview</p>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowFilePreview(!showFilePreview);
                          }}
                          className="text-xs text-blue-600 hover:text-blue-800 flex items-center"
                        >
                          <Eye className="h-3 w-3 mr-1" />
                          {showFilePreview ? 'Hide' : 'Show'}
                        </button>
                      </div>
                      
                      <AnimatePresence>
                        {showFilePreview && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="text-xs bg-gray-50 p-2 rounded border max-h-20 overflow-auto">
                              {previewData.map((row, i) => (
                                <div key={i} className="font-mono text-xs text-gray-300 truncate">
                                  {row.join(', ')}
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <p className="text-lg font-medium text-gray-700 mb-2">
                    {isDragActive ? 'Drop the file here' : 'Drag & drop a file here'}
                  </p>
                  
                  <p className="text-sm text-gray-500 mb-4">
                    {isUploading 
                      ? 'Uploading file...' 
                      : `Supported formats: ${acceptedFileTypes.join(', ')}`}
                  </p>
                </>
              )}

              {uploadProgress && (
                <div className="w-full max-w-xs mx-auto mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-gray-300">Uploading...</span>
                    <span className="text-xs text-gray-300">{uploadProgress.percentage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <motion.div
                      className="bg-blue-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadProgress.percentage}%` }}
                      transition={{ duration: 0.1 }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {(uploadProgress.loaded / (1024 * 1024)).toFixed(1)} MB of {(uploadProgress.total / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
              )}

              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isUploading}
                onClick={(e) => e.stopPropagation()}
                className="relative overflow-hidden"
              >
                {isUploading ? 'Uploading...' : selectedFile ? 'Upload File' : 'Browse Files'}
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg"
          >
            <p className="text-sm text-red-600 flex items-center">
              <AlertCircle className="h-4 w-4 mr-2 flex-shrink-0" />
              {error}
            </p>
          </motion.div>
        )}
      </div>

      {/* Enhanced file type guidance */}
      <div className="mt-6 space-y-3">
        <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
          <FileText className="h-5 w-5 text-purple-500 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-purple-900">Documents (.pdf, .txt)</p>
            <p className="text-xs text-purple-700">Use for RAG chatbot, summarization, and text classification</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
          <Table className="h-5 w-5 text-green-500 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-green-900">Datasets (.csv, .xlsx)</p>
            <p className="text-xs text-green-700">Use for machine learning training and data analysis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
          <Braces className="h-5 w-5 text-blue-500 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-blue-900">Structured Data (.json)</p>
            <p className="text-xs text-blue-700">Use for structured data processing and analysis</p>
          </div>
        </div>
      </div>
      
      {/* File size and format guidelines */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Upload Guidelines</h4>
        <ul className="text-xs text-black space-y-1">
          <li>• Maximum file size: {(maxFileSize / (1024 * 1024)).toFixed(0)}MB</li>
          <li>• For best ML performance, ensure datasets have clear column headers</li>
          <li>• CSV files should use comma separation and UTF-8 encoding</li>
          <li>• Excel files (.xlsx) should have data in the first sheet</li>
        </ul>
      </div>
    </div>
  );
};

export default FileUploadDropzone;
