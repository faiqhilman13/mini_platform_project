export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:8000/api/v1/upload/', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMsg = 'Upload failed.';
    try {
      const err = await response.json();
      errorMsg = err.message || err.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return await response.json();
} 