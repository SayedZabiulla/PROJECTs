import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import json

# Import custom modules
from model_efficientnet import create_efficientnet_model
from train_utils import (
    EarlyStopping, calculate_class_weights, calculate_metrics,
    plot_training_history, plot_confusion_matrix, save_checkpoint
)
from create_dataset import MRISliceDataset, train_transforms, val_test_transforms

# Paths
PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data/processed")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")
LOGS_PATH = os.path.join(PROJECT_PATH, "logs")

# Create directories
EFFICIENTNET_MODELS_PATH = os.path.join(MODELS_PATH, "efficientnet_b0")
EFFICIENTNET_RESULTS_PATH = os.path.join(RESULTS_PATH, "efficientnet_b0")
os.makedirs(EFFICIENTNET_MODELS_PATH, exist_ok=True)
os.makedirs(EFFICIENTNET_RESULTS_PATH, exist_ok=True)

# Training Configuration (SAME AS OTHERS FOR FAIR COMPARISON)
CONFIG = {
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'patience': 10,
    'num_workers': 4,
    'pretrained': True,
    'freeze_backbone': False,
    'use_class_weights': True,
}

print("="*60)
print("EFFICIENTNET-B0 TRAINING CONFIGURATION")
print("="*60)
for key, value in CONFIG.items():
    print(f"{key}: {value}")
print("="*60)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    
    return epoch_loss, metrics


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VAL]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            # Metrics
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    
    return epoch_loss, metrics


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data splits (SAME AS OTHERS)
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    splits_path = os.path.join(PROCESSED_DATA_PATH, "data_splits.pkl")
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    train_data = splits['train']
    val_data = splits['val']
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = MRISliceDataset(train_data, transform=train_transforms)
    val_dataset = MRISliceDataset(val_data, transform=val_test_transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print(f"✓ DataLoaders created")
    
    # Calculate class weights
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_loader)
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        print(f"✓ Using weighted loss with pos_weight: {pos_weight:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("✓ Using standard BCEWithLogitsLoss")
    
    # Create EfficientNet-B0 model
    print("\n" + "="*60)
    print("CREATING EFFICIENTNET-B0 MODEL")
    print("="*60)
    model = create_efficientnet_model(
        pretrained=CONFIG['pretrained'],
        freeze_backbone=CONFIG['freeze_backbone'],
        device=device
    )
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING EFFICIENTNET-B0 TRAINING")
    print("="*60)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1_score']:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | Val F1:   {val_metrics['f1_score']:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(EFFICIENTNET_MODELS_PATH, 'best_model_loss.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, history, best_model_path
            )
        
        # Save best model (based on F1 score)
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_f1_path = os.path.join(EFFICIENTNET_MODELS_PATH, 'best_model_f1.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, history, best_model_f1_path
            )
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n✓ Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    final_model_path = os.path.join(EFFICIENTNET_MODELS_PATH, 'final_model.pth')
    save_checkpoint(
        model, optimizer, scheduler, epoch, best_val_loss, history, final_model_path
    )
    
    # Save training history
    history_path = os.path.join(EFFICIENTNET_RESULTS_PATH, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to: {history_path}")
    
    # Plot training history
    plot_path = os.path.join(EFFICIENTNET_RESULTS_PATH, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Save configuration
    config_path = os.path.join(EFFICIENTNET_RESULTS_PATH, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("EFFICIENTNET-B0 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation F1 Score: {best_val_f1:.4f}")
    print(f"Models saved in: {EFFICIENTNET_MODELS_PATH}")
    print(f"Results saved in: {EFFICIENTNET_RESULTS_PATH}")


if __name__ == "__main__":
    main()
