import React from 'react';
import UploadPage from './UploadPage';
import { BrowserRouter } from 'react-router-dom'; // Required for proper rendering as app uses routing

export default {
  title: 'Pages/UploadPage',
  component: UploadPage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'The main upload page of the application as seen in the real UI.'
      }
    }
  },
  decorators: [
    (Story) => (
      <BrowserRouter>
        <div className="storybook-wrapper" style={{ padding: '0 20px' }}>
          <Story />
        </div>
      </BrowserRouter>
    ),
  ],
};

// Identical to the real application
export const Default = {}; 