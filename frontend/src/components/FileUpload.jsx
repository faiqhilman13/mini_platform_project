import React, { useRef, useState } from 'react';

/**
 * FileUpload component for uploading files (PDF for now).
 *
 * Props:
 *   onUpload: function(file: File) => void
 *   accept: string (accepted file types, e.g. 'application/pdf')
 */
function FileUpload({ onUpload, accept = 'application/pdf' }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setError('');
    const file = e.target.files[0];
    if (!file) return;
    if (accept && !file.type.match(accept)) {
      setError('Invalid file type. Please upload a PDF.');
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = ''; // Clear the input
      return;
    }
    setSelectedFile(file);
  };

  const handleUploadClick = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }
    setUploading(true);
    setError('');
    try {
      await onUpload(selectedFile);
      setSelectedFile(null); // Clear selected file on successful upload call
      if (fileInputRef.current) fileInputRef.current.value = ''; // Clear the file input
    } catch (err) {
      // Error is expected to be set by the parent component (UploadPage)
      // but we catch it here so `setUploading(false)` runs in finally.
      // We also clear the selected file and input if the parent throws an error back.
      setError(err.message || 'Upload failed. Check console for details.'); 
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = ''; 
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ border: '1px solid #eee', padding: 24, borderRadius: 8, maxWidth: 400, fontFamily: 'Arial, sans-serif' }}>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        disabled={uploading}
        style={{ marginBottom: 12, display: 'block' }}
      />
      <button 
        onClick={handleUploadClick} 
        disabled={uploading || !selectedFile} 
        style={{
          marginRight: 8, 
          padding: '8px 16px', 
          cursor: (uploading || !selectedFile) ? 'not-allowed' : 'pointer',
          backgroundColor: (uploading || !selectedFile) ? '#ccc' : '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '4px'
        }}
      >
        {uploading ? 'Uploading...' : 'Upload'}
      </button>
      {selectedFile && <span style={{ marginLeft: '8px', verticalAlign: 'middle' }}>{selectedFile.name}</span>}
      {error && <div style={{ color: 'red', marginTop: 12, fontSize: '0.9em' }}>{error}</div>}
    </div>
  );
}

export default FileUpload;
