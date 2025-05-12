import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
// import TriggerPage from './pages/TriggerPage';
// import StatusPage from './pages/StatusPage';

function App() {
  return (
    <Router>
      <nav style={{ padding: 16, borderBottom: '1px solid #eee', marginBottom: 24 }}>
        <Link to="/" style={{ marginRight: 16 }}>Upload</Link>
        {/* <Link to="/trigger" style={{ marginRight: 16 }}>Trigger Pipeline</Link>
        <Link to="/status">Status</Link> */}
      </nav>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        {/* <Route path="/trigger" element={<TriggerPage />} />
        <Route path="/status" element={<StatusPage />} /> */}
      </Routes>
    </Router>
  );
}

export default App; 