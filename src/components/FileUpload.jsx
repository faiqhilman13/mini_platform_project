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
      return;
    }
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }
    setUploading(true);
    setError('');
    try {
      await onUpload(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err) {
      setError(err.message || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ border: '1px solid #eee', padding: 24, borderRadius: 8, maxWidth: 400 }}>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        disabled={uploading}
        style={{ marginBottom: 12 }}
      />
      <div>
        <button onClick={handleUpload} disabled={uploading || !selectedFile} style={{ marginRight: 8 }}>
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
        {selectedFile && <span>{selectedFile.name}</span>}
      </div>
      {error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
    </div>
  );
}

export default FileUpload; 