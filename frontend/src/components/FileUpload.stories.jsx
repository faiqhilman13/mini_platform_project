import React from 'react';
import FileUpload from './FileUpload';

// More on how to set up stories at: https://storybook.js.org/docs/react/writing-stories/introduction
export default {
  title: 'Components/FileUpload',
  component: FileUpload,
  tags: ['autodocs'], // Enable Autodocs
  parameters: {
    docs: {
      description: {
        component: 'The file upload component as it appears in the real application.'
      }
    }
  }
};

// More on writing stories with args: https://storybook.js.org/docs/react/writing-stories/args
export const Default = {
  args: {
    onUpload: () => console.log('Upload triggered'),
    accept: 'application/pdf',
    disabled: false
  }
};

// With a file selected (using play)
export const WithFileSelected = {
  args: {
    ...Default.args
  },
  play: async ({ canvasElement }) => {
    // This is a mock implementation since we can't actually select a file in Storybook
    // without user interaction, but we can simulate the component's state visually
    const fileInput = canvasElement.querySelector('input[type="file"]');
    
    // Create a custom event to simulate file selection
    const mockFile = new File(['dummy content'], 'example.pdf', { type: 'application/pdf' });
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(mockFile);
    fileInput.files = dataTransfer.files;
    
    // Dispatch change event
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  }
};

// During uploading state
export const Uploading = {
  args: {
    ...Default.args
  },
  render: (args) => {
    // Create a component that forces the uploading state
    const ForcedUploadingState = () => {
      const [isUploading, setIsUploading] = React.useState(true);
      
      return (
        <FileUpload 
          {...args} 
          onUpload={async () => {
            // Keep it in uploading state
            await new Promise(resolve => setTimeout(resolve, 10000));
          }}
          _forceUploading={isUploading} // This would require a prop to be added to FileUpload
        />
      );
    };
    
    return <ForcedUploadingState />;
  }
};

// With error state
export const WithError = {
  args: {
    ...Default.args,
    onUpload: async () => {
      await new Promise(resolve => setTimeout(resolve, 500));
      throw new Error('Upload failed. Server error (500).');
    }
  },
  play: async ({ canvasElement }) => {
    const fileInput = canvasElement.querySelector('input[type="file"]');
    const mockFile = new File(['dummy content'], 'example.pdf', { type: 'application/pdf' });
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(mockFile);
    fileInput.files = dataTransfer.files;
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    
    // Find and click the upload button
    const uploadButton = canvasElement.querySelector('button');
    uploadButton.click();
  }
};

// A story that demonstrates an error message passed via a simulated failed upload
export const UploadFailedError = {
  args: {
    onUpload: async (file) => {
      // Simulate a delay and then an error
      await new Promise(resolve => setTimeout(resolve, 1000));
      throw new Error('Simulated upload failure!');
    },
  },
  // Interaction to trigger the upload and subsequent error
  play: async ({ canvasElement }) => {
    const { userEvent, canvas } = require('@storybook/testing-library');
    const file = new File(['hello'], 'hello.pdf', { type: 'application/pdf' });
    const fileInput = canvas.getByRole('button').previousSibling; // A bit brittle selector

    // // For some reason, the commented out line below doesn't work.
    // // const fileInput = canvas.getByLabelText(/upload/i, { selector: 'input[type="file"]' });

    await userEvent.upload(fileInput, file);
    const uploadButton = canvas.getByRole('button', { name: /upload/i });
    await userEvent.click(uploadButton);
  },
}; 