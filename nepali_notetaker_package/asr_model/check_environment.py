#!/usr/bin/env python3
"""
Quick check script to verify training environment is ready.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...\n")
    
    missing = []
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'evaluate': 'Evaluate',
        'jiwer': 'JiWER',
        'sklearn': 'scikit-learn',
        'pandas': 'Pandas',
        'torchaudio': 'TorchAudio',
        'soundfile': 'SoundFile'
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    return missing

def check_dataset():
    """Check if dataset is available."""
    print("\nChecking dataset...\n")
    
    from config import TRAINING_CONFIG
    
    csv_path = TRAINING_CONFIG['combined_csv']
    
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset found: {csv_path}")
        print(f"  Total samples: {len(df):,}")
        return True
    else:
        print(f"✗ Dataset not found: {csv_path}")
        print("  Run: python create_dataset_csv.py")
        return False

def check_device():
    """Check available compute device."""
    print("\nChecking compute device...\n")
    
    import torch
    
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) available")
        print("  Your Mac M4 Pro GPU will be used for training!")
        return "mps"
    elif torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("⚠ Only CPU available")
        print("  Training will be slower on CPU")
        return "cpu"

def estimate_training_time(num_samples, device):
    """Estimate training time."""
    print("\nTraining time estimate...\n")
    
    # Rough estimates based on device
    if device == "mps":
        samples_per_hour = 5000  # Conservative estimate for M4 Pro
    elif device == "cuda":
        samples_per_hour = 10000
    else:
        samples_per_hour = 500
    
    from config import TRAINING_CONFIG
    
    epochs = TRAINING_CONFIG['num_epochs']
    total_training_samples = num_samples * epochs
    
    hours = total_training_samples / samples_per_hour
    
    print(f"  Epochs: {epochs}")
    print(f"  Samples per epoch: {num_samples:,}")
    print(f"  Estimated time: ~{hours:.1f} hours")
    
    if hours > 24:
        print(f"                  (~{hours/24:.1f} days)")
    
    print("\n  💡 Tip: Training will happen in the background.")
    print("      You can use your computer for other tasks!")

def main():
    print("="*60)
    print("WHISPER FINE-TUNING - ENVIRONMENT CHECK")
    print("="*60 + "\n")
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements_train.txt")
        return False
    
    # Check dataset
    dataset_ok = check_dataset()
    
    if not dataset_ok:
        print("\n❌ Dataset not ready")
        return False
    
    # Check device
    device = check_device()
    
    # Get sample count
    from config import TRAINING_CONFIG
    import pandas as pd
    df = pd.read_csv(TRAINING_CONFIG['combined_csv'])
    
    # Estimate time
    estimate_training_time(len(df), device)
    
    print("\n" + "="*60)
    print("✅ ENVIRONMENT READY FOR TRAINING!")
    print("="*60 + "\n")
    
    print("To start training, run:")
    print("  python train.py")
    print("")
    
    return True

if __name__ == "__main__":
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    success = main()
    sys.exit(0 if success else 1)
