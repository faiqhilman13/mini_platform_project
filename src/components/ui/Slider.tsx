import React, { useState, useEffect } from 'react';
import { cn } from '../../utils/helpers';

interface SliderProps {
  min: number;
  max: number;
  step?: number;
  defaultValue?: number;
  value?: number;
  label?: string;
  showValue?: boolean;
  onChange?: (value: number) => void;
  className?: string;
}

const Slider = ({
  min,
  max,
  step = 1,
  defaultValue,
  value: controlledValue,
  label,
  showValue = true,
  onChange,
  className,
}: SliderProps) => {
  const [value, setValue] = useState(defaultValue || min);
  
  useEffect(() => {
    if (controlledValue !== undefined) {
      setValue(controlledValue);
    }
  }, [controlledValue]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    setValue(newValue);
    
    if (onChange) {
      onChange(newValue);
    }
  };

  // Calculate the percentage for styling the track
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn('space-y-2', className)}>
      {label && (
        <div className="flex justify-between">
          <label className="block text-sm font-medium text-gray-700">{label}</label>
          {showValue && <span className="text-sm font-medium text-gray-700">{value}</span>}
        </div>
      )}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          className={cn(
            'w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer',
            'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
          )}
          style={{
            background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${percentage}%, #E5E7EB ${percentage}%, #E5E7EB 100%)`
          }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};

export default Slider;
