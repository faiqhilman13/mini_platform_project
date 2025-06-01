import { DatasetProfileSummary, PreprocessingConfig } from '../../types';
import Select from '../ui/Select';
import Input from '../ui/Input';
import Slider from '../ui/Slider';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';

interface PreprocessingFormProps {
  profile: DatasetProfileSummary | null;
  config: PreprocessingConfig;
  onChange: (config: PreprocessingConfig) => void;
  className?: string;
}

const PreprocessingForm = ({
  profile,
  config,
  onChange,
  className
}: PreprocessingFormProps) => {
  const handleChange = <K extends keyof PreprocessingConfig>(
    key: K,
    value: PreprocessingConfig[K]
  ) => {
    onChange({
      ...config,
      [key]: value
    });
  };

  if (!profile) {
    return null;
  }

  const columnOptions = profile.columns?.map(column => ({
    value: column.name,
    label: column.name
  })) || [];

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Data Preprocessing</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Target Column"
              options={columnOptions}
              value={config.targetColumn}
              onChange={(value) => handleChange('targetColumn', value)}
              fullWidth
            />
            
            <Select
              label="Problem Type"
              options={[
                { value: 'CLASSIFICATION', label: 'Classification' },
                { value: 'REGRESSION', label: 'Regression' }
              ]}
              value={config.problemType}
              onChange={(value) => handleChange('problemType', value as 'CLASSIFICATION' | 'REGRESSION')}
              fullWidth
            />
          </div>
          
          <Slider
            label="Train/Test Split"
            min={50}
            max={90}
            step={5}
            value={config.trainTestSplit}
            onChange={(value) => handleChange('trainTestSplit', value)}
          />
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Missing Value Strategy"
              options={[
                { value: 'mean', label: 'Mean (for numeric)' },
                { value: 'median', label: 'Median (for numeric)' },
                { value: 'mode', label: 'Mode (most frequent)' },
                { value: 'constant', label: 'Constant value' },
                { value: 'drop', label: 'Drop rows' }
              ]}
              value={config.missingValueStrategy}
              onChange={(value) => handleChange('missingValueStrategy', value as PreprocessingConfig['missingValueStrategy'])}
              fullWidth
            />
            
            {config.missingValueStrategy === 'constant' && (
              <Input
                label="Constant Value"
                value={config.constantValue || ''}
                onChange={(e) => handleChange('constantValue', e.target.value)}
                fullWidth
              />
            )}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Scaling Method"
              options={[
                { value: 'none', label: 'No Scaling' },
                { value: 'standard', label: 'Standard Scaler (Z-score)' },
                { value: 'minmax', label: 'Min-Max Scaler (0-1)' }
              ]}
              value={config.scaling}
              onChange={(value) => handleChange('scaling', value as PreprocessingConfig['scaling'])}
              fullWidth
            />
            
            <Select
              label="Categorical Encoding"
              options={[
                { value: 'onehot', label: 'One-Hot Encoding' },
                { value: 'label', label: 'Label Encoding' }
              ]}
              value={config.categoricalEncoding}
              onChange={(value) => handleChange('categoricalEncoding', value as PreprocessingConfig['categoricalEncoding'])}
              fullWidth
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Feature Selection
            </label>
            <div className="max-h-48 overflow-y-auto border rounded-md p-2">
              <div className="space-y-2">
                {profile.columns?.map((column) => (
                  <div key={column.name} className="flex items-center">
                    <input
                      type="checkbox"
                      id={`feature-${column.name}`}
                      checked={config.featureSelection.includes(column.name)}
                      onChange={(e) => {
                        const newFeatures = e.target.checked
                          ? [...config.featureSelection, column.name]
                          : config.featureSelection.filter(f => f !== column.name);
                        
                        handleChange('featureSelection', newFeatures);
                      }}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      disabled={column.name === config.targetColumn}
                    />
                    <label
                      htmlFor={`feature-${column.name}`}
                      className={`ml-2 text-sm ${
                        column.name === config.targetColumn ? 'text-gray-400' : 'text-gray-700'
                      }`}
                    >
                      {column.name}
                      {column.name === config.targetColumn && ' (target)'}
                    </label>
                  </div>
                )) || (
                  <div className="text-gray-500 text-sm">No columns available</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PreprocessingForm;
