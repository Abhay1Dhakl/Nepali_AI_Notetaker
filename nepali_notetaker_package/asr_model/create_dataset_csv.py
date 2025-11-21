#!/usr/bin/env python3
"""
Script to create a CSV file from the Nepali ASR dataset.
Reads TSV files and creates a CSV with audio filepaths and transcripts.
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import config
from config import DATA_ROOT, TRAINING_CONFIG


def parse_tsv_file(tsv_path, dataset_base_path):
    """
    Parse a TSV file and return list of (filepath, transcript) tuples.
    
    Args:
        tsv_path: Path to the TSV file
        dataset_base_path: Base path for the audio files (e.g., asr_nepali_0/asr_nepali)
    
    Returns:
        List of tuples: (audio_filepath, transcript)
    """
    data = []
    data_dir = dataset_base_path / "data"
    
    print(f"Processing TSV: {tsv_path}")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            file_hash, _, transcript = parts  # Second column is ignored
            
            # Construct audio file path
            # Format: data/XX/hash.flac where XX is first 2 chars of hash
            # For example: hash "0000c0f43b" -> data/00/0000c0f43b.flac
            folder = file_hash[:2]
            audio_path = data_dir / folder / f"{file_hash}.flac"
            
            # Check if file exists
            if audio_path.exists():
                data.append((str(audio_path.absolute()), transcript))
            else:
                print(f"Warning: Audio file not found: {audio_path}")
    
    return data


def create_combined_csv():
    """
    Create a combined CSV file from all dataset folders.
    """
    all_data = []
    
    # Find all dataset folders (asr_nepali_0, asr_nepali_1, etc.)
    dataset_folders = []
    for item in DATA_ROOT.iterdir():
        if item.is_dir() and item.name.startswith('asr_nepali_'):
            dataset_folders.append(item)
    
    dataset_folders.sort()
    print(f"Found {len(dataset_folders)} dataset folders")
    
    # Process each dataset folder
    for dataset_folder in tqdm(dataset_folders, desc="Processing datasets"):
        try:
            print(f"\nProcessing: {dataset_folder.name}")
            
            # Look for TSV files in the dataset folder
            # TSV files are typically in asr_nepali_X/asr_nepali/transcriptX.tsv
            asr_nepali_subfolder = dataset_folder / "asr_nepali"
            
            if not asr_nepali_subfolder.exists():
                print(f"Warning: No 'asr_nepali' subfolder in {dataset_folder.name}")
                continue
            
            # Find TSV files
            tsv_files = list(asr_nepali_subfolder.glob("*.tsv"))
            
            if not tsv_files:
                print(f"Warning: No TSV files found in {asr_nepali_subfolder}")
                continue
            
            for tsv_file in tsv_files:
                data = parse_tsv_file(tsv_file, asr_nepali_subfolder)
                all_data.extend(data)
                print(f"  Added {len(data)} entries from {tsv_file.name}")
        except (OSError, PermissionError) as e:
            print(f"Error processing {dataset_folder.name}: {e}")
            print(f"Skipping this folder and continuing with others...")
            continue
    
    print(f"\nTotal entries collected: {len(all_data)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=['filepath', 'transcript'])
    
    # Save to CSV
    output_csv = TRAINING_CONFIG['combined_csv']
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nCSV file created: {output_csv}")
    print(f"Total records: {len(df)}")
    
    # Display sample
    print("\nSample entries:")
    print(df.head(10))
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique transcripts: {df['transcript'].nunique()}")
    print(f"  Average transcript length: {df['transcript'].str.len().mean():.2f} characters")
    
    return output_csv


def create_sample_csv(num_samples=100):
    """
    Create a smaller sample CSV for testing purposes.
    
    Args:
        num_samples: Number of samples to include
    """
    # Read the full CSV
    full_csv = TRAINING_CONFIG['combined_csv']
    
    if not full_csv.exists():
        print("Full CSV not found. Please run create_combined_csv() first.")
        return
    
    df = pd.read_csv(full_csv)
    
    # Sample randomly
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    # Save sample
    sample_csv = full_csv.parent / "sample_dataset.csv"
    sample_df.to_csv(sample_csv, index=False, encoding='utf-8')
    
    print(f"Sample CSV created: {sample_csv}")
    print(f"Sample size: {len(sample_df)}")
    
    return sample_csv


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Nepali ASR Dataset CSV Generator")
    print("=" * 60)
    
    # Check if dataset exists
    if not DATA_ROOT.exists():
        print(f"Error: Dataset directory not found: {DATA_ROOT}")
        sys.exit(1)
    
    # Create combined CSV
    output_csv = create_combined_csv()
    
    # Optionally create a sample CSV
    print("\n" + "=" * 60)
    create_sample = input("Create a sample CSV file for testing? (y/n): ").strip().lower()
    if create_sample == 'y':
        sample_size = input("Enter sample size (default: 100): ").strip()
        sample_size = int(sample_size) if sample_size.isdigit() else 100
        create_sample_csv(sample_size)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
