import { AlgorithmOption } from '../types';

export const PIPELINE_TYPES = [
  {
    id: 'PDF_SUMMARIZER',
    name: 'Summarizer',
    description: 'Generate concise summaries from document content',
    icon: 'FileText',
    supportedFileTypes: ['pdf', 'text'],
  },
  {
    id: 'RAG_CHATBOT',
    name: 'RAG Chatbot',
    description: 'Create a chat interface with document context awareness',
    icon: 'MessagesSquare',
    supportedFileTypes: ['pdf', 'text'],
  },
  {
    id: 'ML_TRAINING',
    name: 'ML Training',
    description: 'Train machine learning models on structured datasets',
    icon: 'Braces',
    supportedFileTypes: ['csv', 'xlsx'],
  }
];

export const PIPELINE_STATUS_COLORS = {
  QUEUED: 'bg-amber-100 text-amber-800',
  PROCESSING: 'bg-blue-100 text-blue-800',
  COMPLETED: 'bg-green-100 text-green-800',
  FAILED: 'bg-red-100 text-red-800',
};

export const FILE_TYPE_ICONS = {
  pdf: 'FileText',
  csv: 'Table',
  xlsx: 'Table2',
  text: 'File',
  json: 'Braces',
  unknown: 'File',
};

export const ML_ALGORITHMS: AlgorithmOption[] = [
  {
    name: 'logistic_regression',
    displayName: 'Logistic Regression',
    description: 'A linear model for classification problems',
    problemTypes: ['CLASSIFICATION'],
    hyperparameters: [
      {
        name: 'C',
        displayName: 'Regularization Strength',
        type: 'number',
        default: 1.0,
        min: 0.001,
        max: 10,
        step: 0.001,
        description: 'Inverse of regularization strength; smaller values specify stronger regularization',
      },
      {
        name: 'max_iter',
        displayName: 'Max Iterations',
        type: 'number',
        default: 100,
        min: 10,
        max: 1000,
        step: 10,
        description: 'Maximum number of iterations to converge',
      }
    ]
  },
  {
    name: 'random_forest',
    displayName: 'Random Forest',
    description: 'An ensemble of decision trees for classification or regression',
    problemTypes: ['CLASSIFICATION', 'REGRESSION'],
    hyperparameters: [
      {
        name: 'n_estimators',
        displayName: 'Number of Trees',
        type: 'number',
        default: 100,
        min: 10,
        max: 500,
        step: 10,
        description: 'Number of trees in the forest',
      },
      {
        name: 'max_depth',
        displayName: 'Max Depth',
        type: 'number',
        default: 10,
        min: 1,
        max: 50,
        step: 1,
        description: 'Maximum depth of the tree',
      },
      {
        name: 'min_samples_split',
        displayName: 'Min Samples Split',
        type: 'number',
        default: 2,
        min: 2,
        max: 20,
        step: 1,
        description: 'Minimum samples required to split a node',
      }
    ]
  },
  {
    name: 'linear_regression',
    displayName: 'Linear Regression',
    description: 'A linear approach to modeling the relationship between variables',
    problemTypes: ['REGRESSION'],
    hyperparameters: [
      {
        name: 'fit_intercept',
        displayName: 'Fit Intercept',
        type: 'boolean',
        default: true,
        description: 'Whether to calculate the intercept for this model',
      },
      {
        name: 'normalize',
        displayName: 'Normalize',
        type: 'boolean',
        default: false,
        description: 'Whether to normalize the input variables',
      }
    ]
  },
  {
    name: 'decision_tree',
    displayName: 'Decision Tree',
    description: 'A non-parametric method for classification and regression',
    problemTypes: ['CLASSIFICATION', 'REGRESSION'],
    hyperparameters: [
      {
        name: 'max_depth',
        displayName: 'Max Depth',
        type: 'number',
        default: 10,
        min: 1,
        max: 50,
        step: 1,
        description: 'Maximum depth of the tree',
      },
      {
        name: 'min_samples_split',
        displayName: 'Min Samples Split',
        type: 'number',
        default: 2,
        min: 2,
        max: 20,
        step: 1,
        description: 'Minimum samples required to split a node',
      },
      {
        name: 'criterion',
        displayName: 'Split Criterion',
        type: 'select',
        default: 'gini',
        options: ['gini', 'entropy', 'log_loss'],
        description: 'Function to measure the quality of a split',
      }
    ]
  },
  {
    name: 'svm',
    displayName: 'Support Vector Machine',
    description: 'A powerful classifier that works well with small datasets',
    problemTypes: ['CLASSIFICATION', 'REGRESSION'],
    hyperparameters: [
      {
        name: 'C',
        displayName: 'Regularization Parameter',
        type: 'number',
        default: 1.0,
        min: 0.001,
        max: 10,
        step: 0.001,
        description: 'Penalty parameter of the error term',
      },
      {
        name: 'kernel',
        displayName: 'Kernel Type',
        type: 'select',
        default: 'rbf',
        options: ['linear', 'poly', 'rbf', 'sigmoid'],
        description: 'Specifies the kernel type to be used in the algorithm',
      },
      {
        name: 'gamma',
        displayName: 'Kernel Coefficient',
        type: 'select',
        default: 'scale',
        options: ['scale', 'auto'],
        description: 'Kernel coefficient for RBF, poly and sigmoid kernels',
      }
    ]
  },
  {
    name: 'knn',
    displayName: 'K-Nearest Neighbors',
    description: 'A simple, instance-based learning algorithm',
    problemTypes: ['CLASSIFICATION', 'REGRESSION'],
    hyperparameters: [
      {
        name: 'n_neighbors',
        displayName: 'Number of Neighbors',
        type: 'number',
        default: 5,
        min: 1,
        max: 20,
        step: 1,
        description: 'Number of neighbors to use',
      },
      {
        name: 'weights',
        displayName: 'Weight Function',
        type: 'select',
        default: 'uniform',
        options: ['uniform', 'distance'],
        description: 'Weight function used in prediction',
      },
      {
        name: 'algorithm',
        displayName: 'Algorithm',
        type: 'select',
        default: 'auto',
        options: ['auto', 'ball_tree', 'kd_tree', 'brute'],
        description: 'Algorithm used to compute the nearest neighbors',
      }
    ]
  }
];