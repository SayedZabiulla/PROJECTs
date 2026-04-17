import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, ScaleIntensity, Resize, RandAffine, RandFlip
)

class MRISliceDataset(Dataset):
    """Custom Dataset for MRI 2D slices"""
    
    def __init__(self, slice_data, transform=None):
        """
        Args:
            slice_data: List of dicts with 'slice_path' and 'label'
            transform: MONAI transforms to apply
        """
        self.slice_data = slice_data
        self.transform = transform
    
    def __len__(self):
        return len(self.slice_data)
    
    def __getitem__(self, idx):
        # Get slice info
        slice_info = self.slice_data[idx]
        slice_path = slice_info['slice_path']
        label = slice_info['label']
        
        # Load slice and ensure it's 2D
        slice_2d = np.load(slice_path)
        
        # Remove any singleton dimensions and ensure 2D
        slice_2d = np.squeeze(slice_2d)
        
        if slice_2d.ndim != 2:
            raise ValueError(f"Expected 2D slice, got shape {slice_2d.shape} from {slice_path}")
        
        # Add channel dimension: (H, W) -> (1, H, W)
        slice_2d = np.expand_dims(slice_2d, axis=0).astype(np.float32)
        
        # Apply transforms (keep as numpy - MONAI handles this better)
        if self.transform:
            slice_2d = self.transform(slice_2d)
        
        # Convert to tensor
        slice_tensor = torch.as_tensor(slice_2d)
        
        return slice_tensor, label


# Paths
PROCESSED_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/processed")
METADATA_PATH = os.path.join(PROCESSED_DATA_PATH, "slice_metadata.pkl")

# Load metadata
print("Loading slice metadata...")
with open(METADATA_PATH, 'rb') as f:
    slice_metadata = pickle.load(f)

print(f"Total slices: {len(slice_metadata)}")

# Split data: 80% train, 10% val, 10% test
subjects = list(set([m['subject_id'] for m in slice_metadata]))
labels_per_subject = {s: next(m['label'] for m in slice_metadata if m['subject_id'] == s) for s in subjects}

train_subjects, temp_subjects = train_test_split(
    subjects, test_size=0.2, random_state=42, 
    stratify=[labels_per_subject[s] for s in subjects]
)
val_subjects, test_subjects = train_test_split(
    temp_subjects, test_size=0.5, random_state=42,
    stratify=[labels_per_subject[s] for s in temp_subjects]
)

# Create data splits
train_data = [m for m in slice_metadata if m['subject_id'] in train_subjects]
val_data = [m for m in slice_metadata if m['subject_id'] in val_subjects]
test_data = [m for m in slice_metadata if m['subject_id'] in test_subjects]

print(f"\nData Split:")
print(f"Train: {len(train_data)} slices from {len(train_subjects)} subjects")
print(f"Val: {len(val_data)} slices from {len(val_subjects)} subjects")
print(f"Test: {len(test_data)} slices from {len(test_subjects)} subjects")

# Save splits
splits = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}
splits_path = os.path.join(PROCESSED_DATA_PATH, "data_splits.pkl")
with open(splits_path, 'wb') as f:
    pickle.dump(splits, f)

print(f"\n✓ Data splits saved to: {splits_path}")

# Define transforms - Fixed for 2D data
train_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize(spatial_size=(224, 224)),  # Explicitly use spatial_size parameter
    RandAffine(rotate_range=0.1, prob=0.5, padding_mode='zeros'),  # Use RandAffine for 2D rotation
    RandFlip(spatial_axis=0, prob=0.5),
])

val_test_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize(spatial_size=(224, 224)),
])

# Create datasets
print("\nCreating PyTorch Datasets...")
train_dataset = MRISliceDataset(train_data, transform=train_transforms)
val_dataset = MRISliceDataset(val_data, transform=val_test_transforms)
test_dataset = MRISliceDataset(test_data, transform=val_test_transforms)

# Create dataloaders
BATCH_SIZE = 16
NUM_WORKERS = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"\n✓ DataLoaders created")
print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Test loading one batch
print("\nTesting data loading...")
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels: {labels.tolist()}")
    break

print("\n✓ Phase 1 Complete!")
