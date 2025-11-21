# Whisper Fine-Tuning for Nepali ASR

## 📊 Training Overview

This project fine-tunes OpenAI's Whisper model for Nepali automatic speech recognition (ASR) using your Mac M4 Pro.

### Dataset
- **Total Samples**: 157,900 audio-transcript pairs
- **Dataset File**: `aaa_combined_dataset.csv`
- **Language**: Nepali (नेपाली)
- **Audio Format**: FLAC, 16kHz

### Model Configuration
- **Base Model**: `openai/whisper-small`
- **Target Language**: Nepali
- **Task**: Transcription
- **Device**: MPS (Metal Performance Shaders) on Mac M4 Pro

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 4 (effective batch size: 16 with gradient accumulation)
- **Learning Rate**: 1e-5
- **Optimizer**: AdamW
- **Gradient Accumulation Steps**: 4
- **Warmup Steps**: 500
- **Max Gradient Norm**: 1.0

### Estimated Training Time
- **~95 hours** (~4 days)
- Training runs in the background - you can use your computer normally!

## 🚀 How to Use

### 1. Check Environment
```bash
cd nepali_notetaker_package/asr_model
python check_environment.py
```

### 2. Start Training
```bash
python train.py
```

### 3. Monitor Progress
The training script will display:
- Current epoch progress
- Training loss
- Validation loss
- Learning rate
- Time per epoch

### 4. Training Output
Models and checkpoints will be saved to:
```
nepali_notetaker_package/checkpoints/
├── best_model/          # Best performing model
├── checkpoint-step-500/ # Periodic checkpoints
├── checkpoint-step-1000/
└── ...
```

## 📈 What Happens During Training

1. **Data Loading**: Dataset is split into train (90%) and validation (10%)
2. **Training Loop**: Model learns from audio-transcript pairs
3. **Validation**: Model is evaluated after each epoch
4. **Checkpointing**: Best model is saved automatically
5. **Early Stopping**: Training stops if validation loss doesn't improve

## 🛑 Stopping Training

If you need to stop training:
- Press `Ctrl+C` (the current checkpoint will be savedautomatically)
- The best model so far will be in `checkpoints/best_model/`

## 📝 Using the Fine-Tuned Model

After training, use your fine-tuned model:

```python
from asr_model.model import NepaliASR

# Load your fine-tuned model
asr = NepaliASR()  # Will automatically use the best model
transcript = asr.transcribe("path/to/nepali_audio.wav")
print(transcript)
```

## 💡 Tips

1. **Let it run overnight**: Training takes ~4 days, perfect to run when you're not using the computer intensively
2. **Power settings**: Keep your Mac plugged in and prevent sleep mode
3. **Monitor progress**: Check the terminal occasionally to see training progress
4. **Checkpoints**: If training is interrupted, you can resume from the last checkpoint

## 📊 Expected Results

After fine-tuning, you should see:
- **Improved accuracy** for Nepali speech recognition
- **Lower Word Error Rate (WER)** compared to the base model
- **Better understanding** of Nepali-specific vocabulary and accents

## 🔧 Troubleshooting

### Out of Memory
If you encounter memory errors:
1. Reduce `batch_size` in `config.py` (try 2 instead of 4)
2. Reduce `gradient_accumulation_steps`

### Training Too Slow
- This is expected with 157,900 samples
- Consider training on a smaller subset first to test

### Model Not Improving
- Check if validation loss is decreasing
- Early stopping will activate if no improvement after 3 epochs

## 📚 Files Overview

- `train.py` - Main training script
- `data_loader.py` - Dataset and dataloader classes
- `utils.py` - Utility functions (metrics, checkpointing, etc.)
- `config.py` - Training configuration
- `check_environment.py` - Pre-training environment check
- `create_dataset_csv.py` - Dataset preparation script

## 🎯 Next Steps

After training completes:
1. Test the fine-tuned model on real Nepali audio
2. Compare performance with the base Whisper model
3. Iterate on the training (adjust hyperparameters if needed)
4. Deploy the model in your application

---

**Good luck with your training! 🚀**
