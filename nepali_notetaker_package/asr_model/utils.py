#!/usr/bin/env python3
"""
Utility functions for Whisper fine-tuning.
"""

import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_metrics(pred, tokenizer):
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) metrics.
    
    Args:
        pred: Predictions from the model
        tokenizer: Whisper tokenizer
    
    Returns:
        Dictionary with metrics
    """
    import evaluate
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {
        "wer": wer,
        "cer": cer
    }


def setup_device():
    """
    Setup the appropriate device (MPS for Mac M4 Pro, CUDA, or CPU).
    
    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) on Mac M4 Pro")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA on {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def save_checkpoint(
    model,
    processor,
    optimizer,
    epoch: int,
    step: int,
    loss: float,
    metrics: Dict,
    checkpoint_dir: Path,
    is_best: bool = False
):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        processor: The processor to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint metadata
    checkpoint_info = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save model and processor
    checkpoint_name = f"checkpoint-epoch-{epoch}-step-{step}"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    model.save_pretrained(checkpoint_path)
    processor.save_pretrained(checkpoint_path)
    
    # Save optimizer state
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "checkpoint_info": checkpoint_info
        },
        checkpoint_path / "optimizer.pt"
    )
    
    # Save metadata
    with open(checkpoint_path / "checkpoint_info.json", "w") as f:
        json.dump(checkpoint_info, f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as best model if applicable
    if is_best:
        best_path = checkpoint_dir / "best_model"
        model.save_pretrained(best_path)
        processor.save_pretrained(best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model,
    processor,
    optimizer=None
):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load weights into
        processor: Processor to load
        optimizer: Optional optimizer to load state into
    
    Returns:
        checkpoint_info dictionary
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    processor = WhisperProcessor.from_pretrained(checkpoint_path)
    
    # Load optimizer if provided
    if optimizer is not None:
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint_info = checkpoint["checkpoint_info"]
        else:
            logger.warning(f"No optimizer state found at {optimizer_path}")
            checkpoint_info = {}
    else:
        checkpoint_info = {}
    
    # Load metadata if available
    info_path = checkpoint_path / "checkpoint_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            checkpoint_info.update(json.load(f))
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return model, processor, optimizer, checkpoint_info


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered!")
                return True
        
        return False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_training_summary(
    total_epochs: int,
    train_samples: int,
    val_samples: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device
):
    """Print a summary of training configuration."""
    print("\n" + "="*60)
    print("WHISPER FINE-TUNING CONFIGURATION")
    print("="*60)
    print(f"Total Epochs:        {total_epochs}")
    print(f"Training Samples:    {train_samples:,}")
    print(f"Validation Samples:  {val_samples:,}")
    print(f"Batch Size:          {batch_size}")
    print(f"Learning Rate:       {learning_rate}")
    print(f"Device:              {device}")
    print("="*60 + "\n")
