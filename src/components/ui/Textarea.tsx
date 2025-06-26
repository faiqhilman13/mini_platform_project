import React from 'react';
import { cn } from '../../utils/helpers';

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  className?: string;
  fullWidth?: boolean;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ 
    className, 
    label, 
    error, 
    fullWidth = false,
    id,
    ...props 
  }, ref) => {
    const textareaId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className={cn(fullWidth ? 'w-full' : '', className)}>
        {label && (
          <label 
            htmlFor={textareaId} 
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            {label}
          </label>
        )}
        <div className="relative">
          <textarea
            id={textareaId}
            ref={ref}
            className={cn(
              'block rounded-md shadow-sm border-gray-300 dark:border-gray-600',
              'focus:border-blue-500 focus:ring-blue-500',
              'disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed',
              'bg-white dark:bg-gray-700 text-gray-900 dark:text-white',
              'px-4 py-2',
              error ? 'border-red-300' : 'border-gray-300 dark:border-gray-600',
              fullWidth ? 'w-full' : '',
            )}
            {...props}
          />
        </div>
        {error && (
          <p className="mt-1 text-sm text-red-600">{error}</p>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

export default Textarea; 