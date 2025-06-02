import React, { useState } from 'react';
import { FileText, Table, File, ArrowRight, Trash2 } from 'lucide-react';
import { formatBytes, formatDate, cn } from '../../utils/helpers';
import { UploadedFile } from '../../types';
import { FILE_TYPE_ICONS } from '../../utils/constants';
import { Card, CardContent } from '../ui/Card';
import Button from '../ui/Button';

interface FileCardProps {
  file: UploadedFile;
  onClick?: (file: UploadedFile) => void;
  onDelete?: (file: UploadedFile) => void;
  selected?: boolean;
  className?: string;
}

const FileCard = ({ 
  file, 
  onClick, 
  onDelete,
  selected = false,
  className 
}: FileCardProps) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Get the icon component based on file type
  const IconComponent = (() => {
    const iconName = FILE_TYPE_ICONS[file.file_type as keyof typeof FILE_TYPE_ICONS] || 'File';
    
    switch (iconName) {
      case 'FileText': return FileText;
      case 'Table': return Table;
      case 'Table2': return Table;
      default: return File;
    }
  })();

  const handleClick = () => {
    if (onClick) onClick(file);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (showDeleteConfirm) {
      if (onDelete) onDelete(file);
      setShowDeleteConfirm(false);
    } else {
      setShowDeleteConfirm(true);
    }
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowDeleteConfirm(false);
  };

  return (
    <Card 
      className={cn(
        'transition-all duration-200 cursor-pointer',
        selected ? 'ring-2 ring-blue-500 shadow-md' : 'hover:shadow-md',
        className
      )}
      hoverable
    >
      <div onClick={handleClick}>
        <CardContent className="flex items-center p-4">
          <div className={cn(
            'flex items-center justify-center w-12 h-12 rounded-lg',
            file.file_type === 'pdf' || file.file_type === 'text' ? 'bg-purple-100' : 'bg-green-100'
          )}>
            <IconComponent className={cn(
              'w-6 h-6',
              file.file_type === 'pdf' || file.file_type === 'text' ? 'text-purple-600' : 'text-green-600'
            )} />
          </div>
          
          <div className="ml-4 flex-1 min-w-0">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white truncate">
              {file.filename}
            </h3>
            <div className="flex items-center mt-1 text-xs text-gray-500">
              <span>{formatBytes(file.size_bytes)}</span>
              <span className="mx-1">•</span>
              <span>{formatDate(file.upload_timestamp)}</span>
            </div>
          </div>
          
          <div className="ml-4 flex items-center space-x-2">
            {showDeleteConfirm ? (
              <>
                <Button 
                  size="sm" 
                  variant="danger"
                  className="text-xs"
                  onClick={handleDelete}
                >
                  Confirm
                </Button>
                <Button 
                  size="sm" 
                  variant="outline"
                  className="text-xs"
                  onClick={handleCancelDelete}
                >
                  Cancel
                </Button>
              </>
            ) : (
              <>
                {onDelete && (
                  <Button 
                    size="sm" 
                    variant="ghost"
                    className="text-red-400 hover:text-red-600 hover:bg-red-50"
                    onClick={handleDelete}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
                <Button 
                  size="sm" 
                  variant="ghost"
                  className="text-gray-400 hover:text-gray-700"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClick();
                  }}
                >
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </div>
    </Card>
  );
};

export default FileCard;
