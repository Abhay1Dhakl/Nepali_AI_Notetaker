from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from .config import MODEL_PATH

class NepaliASR:
    def __init__(self, model_path=None):
        # Use provided model path or default from config
        load_path = model_path if model_path else MODEL_PATH
        self.processor = WhisperProcessor.from_pretrained(load_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(load_path)
        # Disable forced decoder IDs that cause generate() to fail
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = None
        self.model.generation_config.begin_suppress_tokens = None
    
    def transcribe(self, audio_path: str) -> str:
        waveform, sr = torchaudio.load(audio_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        inputs = self.processor(waveform.squeeze().numpy(), sample_rate=16000, return_tensors="pt")

        # Important fix: avoid forced_decoder_ids error
        predicted_ids = self.model.generate(inputs.input_features, language="ne", 
            task="transcribe", 
            suppress_tokens=[], 
            no_repeat_ngram_size=2 )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
