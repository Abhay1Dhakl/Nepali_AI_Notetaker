#!/usr/bin/env python3
"""
Main training script for fine-tuning Whisper on Nepali ASR dataset.
Optimized for Mac M4 Pro with MPS (Metal Performance Shaders).
"""

import os
import sys
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import time

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup
)

# Import local modules
from config import TRAINING_CONFIG, DATA_ROOT, PACKAGE_ROOT
from data_loader import create_dataloaders
from utils import (
    setup_device,
    save_checkpoint,
    EarlyStopping,
    format_time,
    print_training_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperTrainer:
    """Trainer class for fine-tuning Whisper model."""
    
    def __init__(self, config: dict):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = setup_device()
        
        # Initialize model and processor
        logger.info(f"Loading Whisper model: {config['model_name']}")
        self.processor = WhisperProcessor.from_pretrained(
            config['model_name'],
            language="Nepali",
            task="transcribe"
        )
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config['model_name']
        )
        
        # Configure model for Nepali
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.generation_config.language = "ne"
        self.model.generation_config.task = "transcribe"
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training components
        self.setup_training()
        
    def setup_training(self):
        """Setup optimizer, scheduler, and data loaders."""
        # Create data loaders
        logger.info("Creating data loaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            csv_path=self.config['combined_csv'],
            processor=self.processor,
            train_batch_size=self.config['batch_size'],
            val_batch_size=self.config['batch_size'] * 2,
            num_workers=self.config['dataloader_num_workers']
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * self.config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            mode='min'
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training setup complete!")
        print_training_summary(
            total_epochs=self.config['num_epochs'],
            train_samples=len(self.train_loader.dataset),
            val_samples=len(self.val_loader.dataset),
            batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            device=self.device
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_features=input_features,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item()
            epoch_steps += 1
            
            # Update progress bar
            avg_loss = total_loss / epoch_steps
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Save checkpoint periodically
            if self.global_step % self.config['save_steps'] == 0:
                self.save_checkpoint(is_best=False)
        
        avg_train_loss = total_loss / epoch_steps
        return avg_train_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False
        )
        
        for batch in progress_bar:
            # Move batch to device
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_features=input_features,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
            
            # Update progress bar
            avg_loss = total_loss / total_steps
            progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        avg_val_loss = total_loss / total_steps
        return avg_val_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save a checkpoint."""
        checkpoint_name = f"checkpoint-step-{self.global_step}"
        checkpoint_path = self.output_dir / checkpoint_name
        
        # Save model and processor
        self.model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        
        # Save training state
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path / 'training_state.pt')
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model"
            self.model.save_pretrained(best_path)
            self.processor.save_pretrained(best_path)
            logger.info(f"✓ Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        training_start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log results
            logger.info(
                f"\nEpoch {epoch + 1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Time: {format_time(epoch_time)}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                logger.info(f"✓ New best validation loss: {val_loss:.4f} (previous: {self.best_val_loss:.4f})")
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            # Check early stopping
            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered. Training stopped.")
                break
        
        # Training complete
        total_time = time.time() - training_start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete!")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Model saved to: {self.output_dir / 'best_model'}")
        logger.info(f"{'='*60}\n")


def main():
    """Main function to start training."""
    print("\n" + "="*60)
    print("🎯 WHISPER FINE-TUNING FOR NEPALI ASR")
    print("="*60 + "\n")
    
    # Check if dataset exists
    csv_path = TRAINING_CONFIG['combined_csv']
    if not csv_path.exists():
        logger.error(f"Dataset not found: {csv_path}")
        logger.error("Please run create_dataset_csv.py first!")
        sys.exit(1)
    
    # Check dependencies
    try:
        import evaluate
        import jiwer
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install training requirements:")
        logger.error("pip install -r requirements_train.txt")
        sys.exit(1)
    
    # Create trainer and start training
    try:
        trainer = WhisperTrainer(TRAINING_CONFIG)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
