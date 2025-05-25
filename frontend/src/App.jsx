import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import TriggerPage from './pages/TriggerPage';
import StatusPage from './pages/StatusPage';
import ChatPage from './pages/ChatPage';
import './App.css'; // Assuming you might have some global styles here

function App() {
  return (
    <Router>
      <nav style={{ padding: 16, borderBottom: '1px solid #eee', marginBottom: 24, backgroundColor: '#f8f9fa' }}>
        <Link to="/" style={{ marginRight: 16, textDecoration: 'none', color: '#007bff', fontWeight: 'bold' }}>Upload</Link>
        <Link to="/trigger" style={{ marginRight: 16, textDecoration: 'none', color: '#007bff', fontWeight: 'bold' }}>Trigger Pipeline</Link>
        <Link to="/status" style={{ textDecoration: 'none', color: '#007bff', fontWeight: 'bold' }}>View Status</Link>
      </nav>
      <div style={{ padding: '0 20px' }}>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/trigger" element={<TriggerPage />} />
          <Route path="/status" element={<StatusPage />} />
          <Route path="/status/:runId" element={<StatusPage />} />
          <Route path="/chat/:runId" element={<ChatPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
