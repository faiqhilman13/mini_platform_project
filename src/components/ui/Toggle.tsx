import React from 'react';
import { cn } from '../../utils/helpers';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
}

const Toggle = ({ 
  checked, 
  onChange, 
  label, 
  disabled = false, 
  className 
}: ToggleProps) => {
  return (
    <label className={cn('inline-flex items-center cursor-pointer', disabled && 'opacity-50 cursor-not-allowed', className)}>
      <div className="relative">
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <div
          className={cn(
            'block w-10 h-6 rounded-full transition-colors',
            checked ? 'bg-blue-600' : 'bg-gray-300'
          )}
        ></div>
        <div
          className={cn(
            'dot absolute left-1 top-1 w-4 h-4 rounded-full transition-transform bg-white',
            checked && 'transform translate-x-4'
          )}
        ></div>
      </div>
      {label && <span className="ml-3 text-sm font-medium text-gray-700">{label}</span>}
    </label>
  );
};

export default Toggle;
