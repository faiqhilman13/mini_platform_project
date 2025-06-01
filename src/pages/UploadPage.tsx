import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Clock, FileText, MessageSquare, Tag, Brain } from 'lucide-react';
import FileUploadDropzone from '../components/file/FileUploadDropzone';
import RecentFilesList from '../components/file/RecentFilesList';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { uploadFile, getUploadedFiles } from '../services/api';
import { UploadedFile } from '../types';
import { motion } from 'framer-motion';

const UploadPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchFiles = useCallback(async () => {
    try {
      setIsLoading(true);
      const uploadedFiles = await getUploadedFiles();
      setFiles(uploadedFiles);
      setError(null);
    } catch (err) {
      setError('Failed to load uploaded files. Please try again.');
      console.error('Error fetching files:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleUpload = async (file: File) => {
    try {
      const uploadedFile = await uploadFile(file);
      setFiles(prev => [uploadedFile, ...prev]);
      
      // Navigate to file details page
      navigate(`/files/${uploadedFile.id}`);
    } catch (err) {
      throw new Error('Upload failed. Please try again.');
    }
  };

  const handleFileSelect = (file: UploadedFile) => {
    navigate(`/files/${file.id}`);
  };

  return (
    <div className="px-4 py-6 sm:px-0">
      <div className="space-y-8">
        <div className="text-center max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">
              Mini IDP - AI Workflow Platform
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
              Upload documents and datasets for AI processing, analysis, and model training
            </p>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="lg:col-span-2"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Upload className="h-5 w-5 mr-2 text-blue-600" />
                  Upload Files
                </CardTitle>
              </CardHeader>
              <CardContent>
                <FileUploadDropzone 
                  onUpload={handleUpload}
                  className="mb-4"
                />
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Clock className="h-5 w-5 mr-2 text-purple-600" />
                  Recent Files
                </CardTitle>
              </CardHeader>
              <CardContent>
                <RecentFilesList
                  files={files.slice(0, 5)}
                  isLoading={isLoading}
                  error={error}
                  onFileSelect={handleFileSelect}
                />
                
                {files.length > 5 && (
                  <div className="mt-4 text-center">
                    <button
                      onClick={() => navigate('/documents')}
                      className="text-sm text-blue-600 hover:text-blue-800 flex items-center justify-center"
                    >
                      <FileText className="h-4 w-4 mr-1" />
                      View all files
                    </button>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Available Pipelines</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="flex flex-col">
                  <div className="text-blue-600 mb-3">
                    <MessageSquare className="h-8 w-8" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">RAG Chatbot</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Ask questions about your documents with AI-powered context
                  </p>
                </div>
                <div className="flex flex-col">
                  <div className="text-green-600 mb-3">
                    <FileText className="h-8 w-8" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Summarizer</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Generate concise summaries of long documents
                  </p>
                </div>
                <div className="flex flex-col">
                  <div className="text-purple-600 mb-3">
                    <Tag className="h-8 w-8" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Text Classifier</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Automatically categorize documents by content type
                  </p>
                </div>
                <div className="flex flex-col">
                  <div className="text-blue-600 mb-3">
                    <Brain className="h-8 w-8" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">ML Training</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Train machine learning models on your datasets
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default UploadPage;
