import os
from pathlib import Path

# Get package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DATA_ROOT = PACKAGE_ROOT / "DataSet"

# Model paths
MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    PACKAGE_ROOT / "pretrained" / "openai_nepali_whisper"
)

# Training configuration
TRAINING_CONFIG = {
    # Model settings
    "model_size": "small",  # Options: tiny, base, small, medium, large
    "model_name": "openai/whisper-small",  # Hugging Face model name
    
    # Data settings
    "csv_dir": DATA_ROOT / "Preprocessed_data",
    "combined_csv": DATA_ROOT / "Preprocessed_data" / "aaa_combined_dataset.csv",
    "max_audio_length": 30.0,  # seconds
    "sample_rate": 16000,
    
    # Training hyperparameters
    "learning_rate": 1e-5,
    "num_epochs": 3,
    "batch_size": 4,  # Adjust based on M4 Pro memory
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    
    # Optimization for Mac M4 Pro
    "use_mps": True,  # Metal Performance Shaders for Apple Silicon
    "fp16": False,  # MPS doesn't fully support fp16 yet
    "dataloader_num_workers": 4,  # Adjust based on CPU cores
    
    # Checkpointing
    "output_dir": PACKAGE_ROOT / "checkpoints",
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 3,  # Keep only best 3 checkpoints
    
    # Early stopping
    "early_stopping_patience": 3,
    "metric_for_best_model": "wer",  # Word Error Rate
    "greater_is_better": False,  # Lower WER is better
}
