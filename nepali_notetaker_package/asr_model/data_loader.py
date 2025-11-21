#!/usr/bin/env python3
"""
Data loader for Whisper fine-tuning on Nepali ASR dataset.
Optimized for Mac M4 Pro with MPS (Metal Performance Shaders).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from config import TRAINING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NepaliWhisperDataset(Dataset):
    """Custom Dataset for Nepali ASR data."""
    
    def __init__(
        self,
        csv_path: Path,
        processor: WhisperProcessor,
        max_audio_length: float = 30.0,
        sample_rate: int = 16000,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with 'filepath' and 'transcript' columns
            processor: WhisperProcessor for feature extraction
            max_audio_length: Maximum audio length in seconds
            sample_rate: Target sample rate (16kHz for Whisper)
            split: Dataset split ('train', 'val', 'test')
        """
        self.csv_path = csv_path
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.split = split
        
        # Load CSV
        logger.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Verify all audio files exist
        valid_indices = []
        for idx, row in self.df.iterrows():
            if Path(row['filepath']).exists():
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Loaded {len(self.df)} valid audio samples for {split}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.df.iloc[idx]
        audio_path = row['filepath']
        transcript = row['transcript']
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Truncate or pad to max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Extract features
        audio_array = waveform.squeeze().numpy()
        inputs = self.processor.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Tokenize transcript
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
            padding="max_length",
            max_length=448,  # Whisper's max token length
            truncation=True
        ).input_ids
        
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0)
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text models."""
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )
        
        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove decoder_start_token_id if exists
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


def create_dataloaders(
    csv_path: Path,
    processor: WhisperProcessor,
    train_batch_size: int = 4,
    val_batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.9
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        csv_path: Path to combined CSV file
        processor: WhisperProcessor
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of dataloader workers
        train_split: Fraction of data for training
    
    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    df = pd.read_csv(csv_path)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        train_size=train_split,
        random_state=42,
        shuffle=True
    )
    
    # Save splits
    train_csv = csv_path.parent / "train_split.csv"
    val_csv = csv_path.parent / "val_split.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = NepaliWhisperDataset(
        train_csv,
        processor,
        max_audio_length=TRAINING_CONFIG['max_audio_length'],
        split="train"
    )
    
    val_dataset = NepaliWhisperDataset(
        val_csv,
        processor,
        max_audio_length=TRAINING_CONFIG['max_audio_length'],
        split="val"
    )
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=False  # MPS doesn't support pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=False
    )
    
    return train_loader, val_loader
