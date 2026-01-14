"""
Training Pipeline for DeepG2P Model
Handles antimicrobial resistance prediction from mass spectrometry data
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

from model import DeepG2P, create_deepg2p_model


class DRIAMSDataset(Dataset):
    """Dataset class for DRIAMS mass spectrometry data."""
    
    def __init__(self, features_path, labels_path):
        """
        Initialize dataset.
        
        Args:
            features_path (str): Path to features .npy file
            labels_path (str): Path to labels .npy file
        """
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features ({len(self.features)}) and labels ({len(self.labels)}) "
                "must have same length"
            )
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.from_numpy(self.labels[idx]).float()
        
        # Ensure x has channel dimension [C, L]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add channel dimension
        
        return x, y


def calculate_pos_weight(labels):
    """
    Calculate positive class weights for imbalanced dataset.
    
    Args:
        labels (np.ndarray): Binary labels array
        
    Returns:
        torch.Tensor: Positive class weights
    """
    labels_array = np.load(labels) if isinstance(labels, (str, Path)) else labels
    
    # Calculate ratio of negative to positive samples
    pos_count = labels_array.sum()
    neg_count = len(labels_array) - pos_count
    
    if pos_count == 0:
        return torch.tensor([1.0])
    
    pos_weight = neg_count / pos_count
    print(f"   Class imbalance ratio: {pos_weight:.2f}:1 (negative:positive)")
    
    return torch.tensor([pos_weight])


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item() * inputs.size(0)
        all_targets.append(targets.cpu().detach().numpy())
        all_predictions.append(torch.sigmoid(outputs).cpu().detach().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate AUPRC (Average Precision)
    try:
        auprc = average_precision_score(all_targets, all_predictions)
    except:
        auprc = 0.0
    
    return epoch_loss, auprc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Accumulate metrics
            running_loss += loss.item() * inputs.size(0)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(torch.sigmoid(outputs).cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    try:
        auprc = average_precision_score(all_targets, all_predictions)
        auroc = roc_auc_score(all_targets, all_predictions)
    except:
        auprc = 0.0
        auroc = 0.0
    
    return epoch_loss, auprc, auroc


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)


def train_deepg2p(
    train_features,
    train_labels,
    val_features,
    val_labels,
    output_dir='results',
    batch_size=32,
    epochs=20,
    learning_rate=1e-4,
    num_workers=4,
    model_size='medium',
    num_antibiotics=10
):
    """
    Main training function for DeepG2P model.
    
    Args:
        train_features (str): Path to training features .npy
        train_labels (str): Path to training labels .npy
        val_features (str): Path to validation features .npy
        val_labels (str): Path to validation labels .npy
        output_dir (str): Directory to save results
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        num_workers (int): Number of data loading workers
        model_size (str): Model size ('small', 'medium', 'large')
        num_antibiotics (int): Number of antibiotics to predict
    """
    print("=" * 70)
    print("DeepG2P Training Pipeline - Antimicrobial Resistance Prediction")
    print("=" * 70)
    
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    models_path = Path('models')
    models_path.mkdir(parents=True, exist_ok=True)
    logs_path = output_path / 'logs'
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=str(logs_path))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create datasets
    print("\n📂 Loading datasets...")
    train_dataset = DRIAMSDataset(train_features, train_labels)
    val_dataset = DRIAMSDataset(val_features, val_labels)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Calculate positive class weights for imbalanced data
    print("\n⚖️  Calculating class weights for imbalanced data...")
    pos_weight = calculate_pos_weight(train_labels).to(device)
    
    # Create model
    print(f"\n🏗️  Building DeepG2P model (size: {model_size})...")
    model = create_deepg2p_model(
        input_length=6000,
        input_channels=1,
        num_antibiotics=num_antibiotics,
        model_size=model_size
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Setup loss function with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Setup optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Optimizer: AdamW")
    print(f"   Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_auprc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_auprc, val_auroc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AUPRC/train', train_auprc, epoch)
        writer.add_scalar('AUPRC/val', val_auprc, epoch)
        writer.add_scalar('AUROC/val', val_auroc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train AUPRC: {train_auprc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val AUPRC:   {val_auprc:.4f} | Val AUROC: {val_auroc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = models_path / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"   ✅ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = models_path / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = models_path / 'final_model.pth'
    save_checkpoint(model, optimizer, epochs, val_loss, final_model_path)
    
    # Save training configuration
    config = {
        'train_features': str(train_features),
        'train_labels': str(train_labels),
        'val_features': str(val_features),
        'val_labels': str(val_labels),
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_size': model_size,
        'num_antibiotics': num_antibiotics,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'pos_weight': pos_weight.item(),
        'training_date': datetime.now().isoformat()
    }
    
    config_path = output_path / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Close tensorboard writer
    writer.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n✅ Best model: {best_model_path}")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"\n📁 Outputs:")
    print(f"   - Best model: {best_model_path}")
    print(f"   - Final model: {final_model_path}")
    print(f"   - Config: {config_path}")
    print(f"   - Tensorboard logs: {logs_path}")
    print(f"\n📈 View training curves:")
    print(f"   tensorboard --logdir {logs_path}")
    print("=" * 70)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train DeepG2P model for antimicrobial resistance prediction'
    )
    
    # Data paths
    parser.add_argument(
        '--train-features',
        type=str,
        default='data/processed/X_train.npy',
        help='Path to training features .npy file'
    )
    parser.add_argument(
        '--train-labels',
        type=str,
        default='data/processed/y_train.npy',
        help='Path to training labels .npy file'
    )
    parser.add_argument(
        '--val-features',
        type=str,
        default='data/processed/X_val.npy',
        help='Path to validation features .npy file'
    )
    parser.add_argument(
        '--val-labels',
        type=str,
        default='data/processed/y_val.npy',
        help='Path to validation labels .npy file'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Model size (default: medium)'
    )
    parser.add_argument(
        '--num-antibiotics',
        type=int,
        default=10,
        help='Number of antibiotics to predict (default: 10)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Run training
    train_deepg2p(
        train_features=args.train_features,
        train_labels=args.train_labels,
        val_features=args.val_features,
        val_labels=args.val_labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        model_size=args.model_size,
        num_antibiotics=args.num_antibiotics
    )


if __name__ == '__main__':
    main()
