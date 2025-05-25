import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import MultiPipelineSelector from './MultiPipelineSelector';

export default {
  title: 'Components/MultiPipelineSelector',
  component: MultiPipelineSelector,
  tags: ['autodocs'],
  parameters: {
    docs: {
      description: {
        component: 'A component for selecting multiple pipelines to run on a document.'
      }
    }
  },
  decorators: [
    (Story) => (
      <BrowserRouter>
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
          <Story />
        </div>
      </BrowserRouter>
    ),
  ],
};

// Default state
export const Default = {
  args: {
    fileLogId: 'file-abc-123',
    onSubmit: (selectedPipelines) => {
      console.log('Selected pipelines:', selectedPipelines);
    },
    disabled: false
  }
};

// Loading state
export const Loading = {
  args: {
    ...Default.args,
    disabled: true
  }
};

// With pre-selected pipelines
export const PreSelected = {
  args: {
    ...Default.args,
    preSelectedPipelines: ['PDF_SUMMARIZER', 'RAG_CHATBOT']
  }
}; 