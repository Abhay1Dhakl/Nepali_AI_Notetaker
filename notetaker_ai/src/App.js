import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import HomePage from './HomePage';
import LivePage from './LivePage';

function App() {
  return (
    <Router>
      <Routes>
        {/* <Route path="/" element={<HomePage />} /> */}
        <Route path="/live" element={<LivePage />} />
      </Routes>
    </Router>
  );
}

export default App;
