import React from 'react';
import './ui_seperation.css';

function ui_seperation(){
    return(
        <div className="App">
      <h1>Nepali Notetaker AI</h1>
      <div className="card-container">
        
        {/* Upload Card */}
        <div className="card">
          <h2>Upload Files</h2>
          <p>Upload Audio, Video, or Image files to generate notes.</p>
          <button>Upload</button>
        </div>

        {/* Live Section Card */}
        <div className="card">
          <h2>Live Section</h2>
          <p>Go to live meeting mode and take notes in real-time.</p>
          <button>Go Live</button>
        </div>

      </div>
    </div>
    )
}

export default ui_seperation