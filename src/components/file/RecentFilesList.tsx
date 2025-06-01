import React from 'react';
import { FileX } from 'lucide-react';
import FileCard from './FileCard';
import { UploadedFile } from '../../types';
import Spinner from '../ui/Spinner';

interface RecentFilesListProps {
  files: UploadedFile[];
  isLoading: boolean;
  error: string | null;
  onFileSelect: (file: UploadedFile) => void;
  selectedFileId?: string;
  className?: string;
}

const RecentFilesList = ({
  files,
  isLoading,
  error,
  onFileSelect,
  selectedFileId,
  className
}: RecentFilesListProps) => {
  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-12">
        <Spinner size="md" />
        <span className="ml-3 text-sm text-gray-300">Loading files...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <FileX className="h-12 w-12 text-red-500 mb-4" />
        <h3 className="text-lg font-medium text-white mb-1">Failed to load files</h3>
        <p className="text-sm text-gray-300 mb-4">{error}</p>
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <FileX className="h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-white mb-1">No files uploaded yet</h3>
        <p className="text-sm text-gray-300">Upload a file to get started</p>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="space-y-3">
        {files.map((file) => (
          <FileCard
            key={file.id}
            file={file}
            onClick={onFileSelect}
            selected={file.id === selectedFileId}
          />
        ))}
      </div>
    </div>
  );
};

export default RecentFilesList;
