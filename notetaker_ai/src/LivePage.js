import React, { useState } from 'react';

const LivePage = () => {
  const [transcript, setTranscript] = useState("");
  const [listening, setListening] = useState(false);

  let recognition;

  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'ne-NP'; // Nepali
    recognition.continuous = true;

    recognition.onresult = (event) => {
      const current = event.resultIndex;
      const result = event.results[current][0].transcript;
      setTranscript(prev => prev + " " + result);
    };

    recognition.start();
    setListening(true);
  };

  const stopListening = () => {
    if (recognition) {
      recognition.stop();
      setListening(false);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h2>🎤 Live Nepali Notetaker</h2>
      <button onClick={startListening} disabled={listening}>Start Listening</button>
      <button onClick={stopListening} disabled={!listening}>Stop Listening</button>

      <div style={{
        marginTop: '2rem',
        padding: '1rem',
        background: '#f9f9f9',
        borderRadius: '10px',
        minHeight: '150px'
      }}>
        <h4>📝 Transcribed Notes:</h4>
        <p>{transcript}</p>
      </div>
    </div>
  );
};

export default LivePage;
