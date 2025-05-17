import React, { useRef, useState, useEffect } from 'react';

/**
 * FileUpload component for uploading files (PDF for now).
 *
 * Props:
 *   onUpload: function(file: File) => void
 *   accept: string (accepted file types, e.g. 'application/pdf')
 *   disabled: boolean (optional)
 *   _forceUploading: boolean (optional, for Storybook)
 *   _forceError: string (optional, for Storybook)
 *   _forceSelectedFile: File (optional, for Storybook)
 */
function FileUpload({ 
  onUpload, 
  accept = 'application/pdf', 
  disabled = false,
  _forceUploading,
  _forceError,
  _forceSelectedFile
}) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(_forceSelectedFile || null);
  const [error, setError] = useState(_forceError || '');
  const [uploading, setUploading] = useState(Boolean(_forceUploading) || false);

  // For Storybook to control component state
  useEffect(() => {
    if (_forceUploading !== undefined) {
      setUploading(Boolean(_forceUploading));
    }
  }, [_forceUploading]);

  useEffect(() => {
    if (_forceError !== undefined) {
      setError(_forceError);
    }
  }, [_forceError]);

  useEffect(() => {
    if (_forceSelectedFile !== undefined) {
      setSelectedFile(_forceSelectedFile);
    }
  }, [_forceSelectedFile]);

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
      // Only clear the uploading state if we're not forcing it for Storybook
      if (_forceUploading === undefined) {
        setUploading(false);
      }
    }
  };

  return (
    <div className="card">
      <div className="form-group">
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileChange}
          disabled={uploading || disabled}
          className="form-control mb-3" 
        />
        
        <div className="d-flex align-items-center">
          <button 
            onClick={handleUploadClick} 
            disabled={uploading || !selectedFile || disabled}
            className="form-control"
          >
            {uploading ? (
              <span><div className="loading"></div> Uploading...</span>
            ) : (
              'Upload'
            )}
          </button>
          
          {selectedFile && (
            <div className="ml-2 mt-2">
              <strong>Selected:</strong> {selectedFile.name}
            </div>
          )}
        </div>
      </div>
      
      {error && (
        <div className="alert alert-danger mt-3">{error}</div>
      )}
    </div>
  );
}

export default FileUpload;
