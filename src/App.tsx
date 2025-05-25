import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Upload, Brain, Settings2, BarChart3 } from 'lucide-react';
import Header from './components/layout/Header';
import UploadPage from './pages/UploadPage';
import DatasetConfigPage from './pages/DatasetConfigPage';
import MLConfigPage from './pages/MLConfigPage';
import MLResultsPage from './pages/MLResultsPage';
import FileDetailsPage from './pages/FileDetailsPage';import ChatPage from './pages/ChatPage';import PipelineResultsPage from './pages/PipelineResultsPage';

// Import proper page components
import FilesListPage from './pages/FilesListPage';
import MLTrainingPage from './pages/MLTrainingPage';
import ResultsListPage from './pages/ResultsListPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            {/* Default route redirects to upload */}
            <Route path="/" element={<Navigate to="/upload" replace />} />
            
            {/* File upload and management */}
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/files" element={<FilesListPage />} />
            <Route path="/files/:fileId" element={<FileDetailsPage />} />
            
            {/* ML Training Routes */}
            <Route path="/ml" element={<MLTrainingPage />} />
            <Route path="/dataset/:fileId/config" element={<DatasetConfigPage />} />
            <Route path="/ml/:fileId/config" element={<MLConfigPage />} />
            
            {/* Results Routes */}
            <Route path="/results" element={<ResultsListPage />} />
            <Route path="/ml/results/:runId" element={<MLResultsPage />} />
            
                        {/* Chat/RAG Interface */}            <Route path="/chat/:runId" element={<ChatPage />} />                        {/* Pipeline Results */}            <Route path="/pipeline-results/:runId" element={<PipelineResultsPage />} />
            
            {/* Catch all - redirect to upload */}
            <Route path="*" element={<Navigate to="/upload" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;