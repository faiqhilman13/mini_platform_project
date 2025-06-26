import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, MessagesSquare, Braces, ArrowRight } from 'lucide-react';
import { PIPELINE_TYPES } from '../../utils/constants';
import { PipelineType, FileType } from '../../types';
import { Card, CardContent, CardFooter } from '../ui/Card';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';

interface PipelineSelectorProps {
  fileType: FileType;
  onSelect: (pipelineType: PipelineType) => void;
  className?: string;
}

const PipelineSelector = ({ 
  fileType, 
  onSelect,
  className 
}: PipelineSelectorProps) => {
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Filter pipeline types based on file type
  const filteredPipelines = PIPELINE_TYPES.filter(pipeline => 
    pipeline.supportedFileTypes.includes(fileType)
  );

  // Get the icon component based on the icon name
  const getIconComponent = (iconName: string) => {
    switch (iconName) {
      case 'FileText': return <FileText className="h-6 w-6" />;
      case 'MessagesSquare': return <MessagesSquare className="h-6 w-6" />;
      case 'Braces': return <Braces className="h-6 w-6" />;
      default: return <FileText className="h-6 w-6" />;
    }
  };

  const handleSelect = (pipelineType: PipelineType) => {
    onSelect(pipelineType);
  };

  if (filteredPipelines.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">No supported pipelines for this file type.</p>
      </div>
    );
  }

  return (
    <div className={cn("grid gap-4 grid-cols-1 md:grid-cols-2", className)}>
      {filteredPipelines.map((pipeline) => (
        <motion.div
          key={pipeline.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          whileHover={{ scale: 1.02 }}
          onMouseEnter={() => setHoveredId(pipeline.id)}
          onMouseLeave={() => setHoveredId(null)}
        >
          <Card 
            className={`h-full ${
              hoveredId === pipeline.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className={cn(
                  "flex items-center justify-center w-12 h-12 rounded-lg",
                  pipeline.id === 'RAG_CHATBOT' ? 'bg-blue-100 text-blue-600' :
                  pipeline.id === 'SUMMARIZER' ? 'bg-purple-100 text-purple-600' :
                  'bg-green-100 text-green-600'
                )}>
                  {getIconComponent(pipeline.icon)}
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">{pipeline.name}</h3>
                </div>
              </div>
              <p className="mt-4 text-sm text-gray-500">{pipeline.description}</p>
            </CardContent>
            <CardFooter className="px-6 py-4 bg-gray-50 border-t">
              <Button 
                onClick={() => handleSelect(pipeline.id as PipelineType)}
                className="w-full"
                variant={hoveredId === pipeline.id ? 'primary' : 'outline'}
              >
                <span>Select Pipeline</span>
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </motion.div>
      ))}
    </div>
  );
};

export default PipelineSelector;
