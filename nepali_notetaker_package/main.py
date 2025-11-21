import argparse
from pathlib import Path
from asr_model.model import NepaliASR

def main():
    parser = argparse.ArgumentParser(description="Nepali ASR - Inference Mode")
    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to audio file for transcription"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (optional)"
    )
    
    args = parser.parse_args()
    
    # Use provided audio path or example from dataset
    if args.audio_path:
        audio_path = Path(args.audio_path)
    else:
        # Try to find a sample audio file from dataset
        dataset_dir = Path(__file__).parent / "DataSet"
        audio_files = list(dataset_dir.rglob("*.wav")) + list(dataset_dir.rglob("*.flac"))
        
        audio_path= audio_files[0]
        print(f"No audio path provided, using sample: {audio_path}")
        
        else:
            print("Error: No audio file found. Please provide --audio_path argument")
            return
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    # Initialize ASR model
    print(f"Loading ASR model...")
    asr = NepaliASR(model_path=args.model_path)
    
    # Transcribe
    print(f"Transcribing: {audio_path}")
    transcript = asr.transcribe(str(audio_path))
    print(f"\nTranscription: {transcript}")

if __name__ == "__main__":
    main()
