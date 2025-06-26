import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Upload, 
  Search, 
  Filter, 
  Grid3X3, 
  List, 
  FileText, 
  Database,
  Brain,
  RefreshCw,
  AlertCircle,
  Bot
} from 'lucide-react';
import { motion } from 'framer-motion';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import Button from '../components/ui/Button';
import FileCard from '../components/file/FileCard';
import { getUploadedFiles, deleteFile } from '../services/api';
import { UploadedFile } from '../types';
import { cn } from '../utils/helpers';

const FilesListPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [filteredFiles, setFilteredFiles] = useState<UploadedFile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'documents' | 'datasets'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [deletingFileId, setDeletingFileId] = useState<string | null>(null);

  const fetchFiles = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const uploadedFiles = await getUploadedFiles();
      setFiles(uploadedFiles);
      setFilteredFiles(uploadedFiles);
    } catch (err) {
      setError('Failed to load files. Please try again.');
      console.error('Error fetching files:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  // Filter and search files
  useEffect(() => {
    let filtered = files;

    // Apply type filter
    if (filterType === 'documents') {
      filtered = filtered.filter(file => ['pdf', 'text'].includes(file.file_type));
    } else if (filterType === 'datasets') {
      filtered = filtered.filter(file => ['csv', 'xlsx'].includes(file.file_type));
    }

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(file =>
        file.filename.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredFiles(filtered);
  }, [files, filterType, searchTerm]);

  const handleFileSelect = (file: UploadedFile) => {
    navigate(`/files/${file.id}`);
  };

  const handleStartMLWorkflow = (file: UploadedFile) => {
    if (['csv', 'xlsx'].includes(file.file_type)) {
      navigate(`/dataset/${file.id}/config`);
    }
  };

  const handleInteractWithLLMs = (file: UploadedFile) => {
    // Navigate to AI page with file context
    navigate('/ai', { state: { selectedFile: file } });
  };

  const handleDeleteFile = async (file: UploadedFile) => {
    try {
      setDeletingFileId(file.id);
      setError(null);
      
      const result = await deleteFile(file.id);
      
      if (result.success) {
        // Remove the file from local state
        setFiles(prevFiles => prevFiles.filter(f => f.id !== file.id));
        setFilteredFiles(prevFiles => prevFiles.filter(f => f.id !== file.id));
      } else {
        setError(`Failed to delete file: ${result.message}`);
      }
    } catch (err) {
      setError('Failed to delete file. Please try again.');
      console.error('Error deleting file:', err);
    } finally {
      setDeletingFileId(null);
    }
  };

  const getFileTypeStats = () => {
    const stats = {
      total: files.length,
      documents: files.filter(f => ['pdf', 'text'].includes(f.file_type)).length,
      datasets: files.filter(f => ['csv', 'xlsx'].includes(f.file_type)).length,
    };
    return stats;
  };

  // Helper function to check if file is a document that can work with LLMs
  const isDocumentFile = (fileType: string) => {
    return ['pdf', 'text'].includes(fileType) || 
           fileType === 'application/pdf' || 
           fileType === 'text/plain' ||
           fileType === 'application/msword' ||
           fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
  };

  const stats = getFileTypeStats();

  if (isLoading) {
    return (
      <div className="px-4 py-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
            <p className="text-gray-300">Loading files...</p>
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
            <h1 className="text-3xl font-bold text-gray-900 flex items-center">
              <Database className="h-8 w-8 text-purple-600 mr-3" />
              Files
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              Manage your uploaded documents and datasets
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              onClick={fetchFiles}
              className="text-sm"
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
            <Button
              onClick={() => navigate('/upload')}
              className="text-sm"
            >
              <Upload className="h-4 w-4 mr-1" />
              Upload Files
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-800 dark:text-gray-300">Total Files</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.total}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <FileText className="h-6 w-6 text-purple-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-800 dark:text-gray-300">Documents</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.documents}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Database className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-800 dark:text-gray-300">Datasets</p>
                  <p className="text-2xl font-bold text-black dark:text-white">{stats.datasets}</p>
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
                  placeholder="Search files..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>

              {/* Filter */}
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-gray-500" />
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value as any)}
                  className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value="all">All Files</option>
                  <option value="documents">Documents</option>
                  <option value="datasets">Datasets</option>
                </select>
              </div>

              {/* View Mode */}
              <div className="flex rounded-lg border border-gray-300">
                <button
                  onClick={() => setViewMode('grid')}
                  className={cn(
                    'p-2 rounded-l-lg transition-colors',
                    viewMode === 'grid' ? 'bg-purple-100 text-purple-600' : 'text-gray-500 hover:text-gray-700'
                  )}
                >
                  <Grid3X3 className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={cn(
                    'p-2 rounded-r-lg transition-colors',
                    viewMode === 'list' ? 'bg-purple-100 text-purple-600' : 'text-gray-500 hover:text-gray-700'
                  )}
                >
                  <List className="h-4 w-4" />
                </button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Files List */}
        {error ? (
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center text-red-500">
                <AlertCircle className="h-5 w-5 mr-2" />
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        ) : filteredFiles.length === 0 ? (
          <Card>
            <CardContent className="p-6">
              <div className="text-center py-8">
                <Database className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {searchTerm || filterType !== 'all' ? 'No files found' : 'No files uploaded'}
                </h3>
                <p className="text-gray-300 mb-4">
                  {searchTerm || filterType !== 'all' 
                    ? 'Try adjusting your search or filter criteria.'
                    : 'Upload your first file to get started with AI workflows.'
                  }
                </p>
                {!searchTerm && filterType === 'all' && (
                  <Button onClick={() => navigate('/upload')}>
                    <Upload className="h-4 w-4 mr-1" />
                    Upload Files
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className={cn(
            viewMode === 'grid' 
              ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
              : 'space-y-3'
          )}>
            {filteredFiles.map((file, index) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <FileCard
                  file={file}
                  onClick={handleFileSelect}
                  onDelete={handleDeleteFile}
                  className={viewMode === 'list' ? 'w-full' : ''}
                />
                
                {/* Action Buttons for different file types */}
                <div className="mt-2 space-y-2">
                  {/* ML Training Button for Datasets */}
                  {['csv', 'xlsx'].includes(file.file_type) && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleStartMLWorkflow(file);
                      }}
                      className="w-full text-xs"
                      disabled={deletingFileId === file.id}
                    >
                      <Brain className="h-3 w-3 mr-1" />
                      Start ML Training
                    </Button>
                  )}
                  
                  {/* LLM Interaction Button for Documents */}
                  {isDocumentFile(file.file_type) && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleInteractWithLLMs(file);
                      }}
                      className="w-full text-xs"
                      disabled={deletingFileId === file.id}
                    >
                      <Bot className="h-3 w-3 mr-1" />
                      Interact with LLMs
                    </Button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default FilesListPage; 
