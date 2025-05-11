import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import { uploadFile } from '../services/api';

function UploadPage() {
  const [success, setSuccess] = useState(false);
  const [fileLogId, setFileLogId] = useState(null);
  const [message, setMessage] = useState('');

  const handleUpload = async (file) => {
    setSuccess(false);
    setFileLogId(null);
    setMessage('');
    try {
      const response = await uploadFile(file);
      setSuccess(true);
      setFileLogId(response.file_log_id);
      setMessage(response.message || 'File uploaded successfully!');
    } catch (err) {
      setSuccess(false);
      setMessage(err.message || 'Upload failed.');
      throw err; // Re-throw to allow FileUpload component to catch it if desired
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 40 }}>
      <h2>Upload a PDF File</h2>
      <FileUpload onUpload={handleUpload} accept="application/pdf" />
      {success && (
        <div style={{ color: 'green', marginTop: 16 }}>
          {message} <br />
          <strong>File Log ID:</strong> {fileLogId}
        </div>
      )}
      {!success && message && (
        <div style={{ color: 'red', marginTop: 16 }}>{message}</div>
      )}
    </div>
  );
}

export default UploadPage;
