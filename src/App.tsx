import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Upload, Brain, Settings2, BarChart3 } from 'lucide-react';
import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/layout/Header';
import HeroPage from './pages/HeroPage';
import UploadPage from './pages/UploadPage';
import DatasetConfigPage from './pages/DatasetConfigPage';
import MLConfigPage from './pages/MLConfigPage';
import MLResultsPage from './pages/MLResultsPage';
import FileDetailsPage from './pages/FileDetailsPage';
import ChatPage from './pages/ChatPage';
import PipelineResultsPage from './pages/PipelineResultsPage';

// Import proper page components
import FilesListPage from './pages/FilesListPage';
import MLTrainingPage from './pages/MLTrainingPage';
import ResultsListPage from './pages/ResultsListPage';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
          <Routes>
            {/* Hero/Landing page as default */}
            <Route path="/" element={<HeroPage />} />
            
            {/* App pages with header */}
            <Route path="/upload" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <UploadPage />
                </main>
              </>
            } />
            <Route path="/files" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <FilesListPage />
                </main>
              </>
            } />
            <Route path="/files/:fileId" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <FileDetailsPage />
                </main>
              </>
            } />
            
            {/* ML Training Routes */}
            <Route path="/ml" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <MLTrainingPage />
                </main>
              </>
            } />
            <Route path="/dataset/:fileId/config" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <DatasetConfigPage />
                </main>
              </>
            } />
            <Route path="/ml/:fileId/config" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <MLConfigPage />
                </main>
              </>
            } />
            
            {/* Results Routes */}
            <Route path="/results" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <ResultsListPage />
                </main>
              </>
            } />
            <Route path="/ml/results/:runId" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <MLResultsPage />
                </main>
              </>
            } />
            
            {/* Chat/RAG Interface */}
            <Route path="/chat/:runId" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <ChatPage />
                </main>
              </>
            } />
            
            {/* Pipeline Results */}
            <Route path="/pipeline-results/:runId" element={
              <>
                <Header />
                <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                  <PipelineResultsPage />
                </main>
              </>
            } />
            
            {/* Catch all - redirect to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
