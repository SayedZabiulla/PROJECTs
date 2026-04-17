import os
import glob
import nibabel as nib
import numpy as np
import json
import pickle
from tqdm import tqdm

# Paths
RAW_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/raw")
PROCESSED_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/processed")
SLICES_PATH = os.path.join(PROCESSED_DATA_PATH, "slices")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_PATH, "label_map.json")

# Parameters
SLICE_AXIS = 1  # 0=sagittal, 1=coronal, 2=axial
SLICE_RANGE = (0.3, 0.7)  # Extract middle 40% of slices

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

# Create slices directory
os.makedirs(SLICES_PATH, exist_ok=True)

# Get all subject folders from all 12 discs
all_subject_folders = []

print("Searching for subjects across all discs...")
for disc_num in range(1, 13):  # disc1 to disc12
    disc_path = os.path.join(RAW_DATA_PATH, f"oasis_cross-sectional_disc{disc_num}/disc{disc_num}")
    
    if os.path.exists(disc_path):
        subject_folders = glob.glob(os.path.join(disc_path, "OAS1_*_MR*"))
        all_subject_folders.extend(subject_folders)
        print(f"  Disc {disc_num}: Found {len(subject_folders)} subjects")

# Sort all subjects
all_subject_folders = sorted(all_subject_folders)
print(f"\n✓ Total subjects found: {len(all_subject_folders)}")

# Slice metadata
slice_metadata = []

print(f"\nExtracting 2D slices from {len(all_subject_folders)} subjects...")
print(f"Slice axis: {SLICE_AXIS} (0=sagittal, 1=coronal, 2=axial)")
print(f"Slice range: {SLICE_RANGE[0]*100}% to {SLICE_RANGE[1]*100}%\n")

for subject_folder in tqdm(all_subject_folders):
    subject_id = os.path.basename(subject_folder)
    
    # Check if subject has label
    if subject_id not in label_map:
        continue
    
    label = label_map[subject_id]
    
    # Find all .img files in the RAW folder (updated path)
    raw_folder = os.path.join(subject_folder, "RAW")
    img_files = glob.glob(os.path.join(raw_folder, "*mpr*.img"))
    
    if len(img_files) == 0:
        continue
    
    # Load and average scans
    scan_data = []
    for img_file in img_files:
        try:
            img = nib.load(img_file)
            data = img.get_fdata()
            scan_data.append(data)
        except:
            continue
    
    if len(scan_data) == 0:
        continue
    
    # Average scans
    averaged_volume = np.mean(scan_data, axis=0).astype(np.float32)
    
    # Extract slices
    num_slices = averaged_volume.shape[SLICE_AXIS]
    start_idx = int(num_slices * SLICE_RANGE[0])
    end_idx = int(num_slices * SLICE_RANGE[1])
    
    for slice_idx in range(start_idx, end_idx):
        # Extract slice
        if SLICE_AXIS == 0:
            slice_2d = averaged_volume[slice_idx, :, :]
        elif SLICE_AXIS == 1:
            slice_2d = averaged_volume[:, slice_idx, :]
        else:  # SLICE_AXIS == 2
            slice_2d = averaged_volume[:, :, slice_idx]
        
        # Skip empty slices
        if slice_2d.max() == 0:
            continue
        
        # Save slice
        slice_filename = f"{subject_id}_slice_{slice_idx:03d}.npy"
        slice_path = os.path.join(SLICES_PATH, slice_filename)
        np.save(slice_path, slice_2d)
        
        # Save metadata
        slice_metadata.append({
            'slice_path': slice_path,
            'subject_id': subject_id,
            'label': label,
            'slice_idx': slice_idx,
            'shape': slice_2d.shape
        })

# Save slice metadata
metadata_path = os.path.join(PROCESSED_DATA_PATH, "slice_metadata.pkl")
with open(metadata_path, 'wb') as f:
    pickle.dump(slice_metadata, f)

print(f"\n✓ Extracted {len(slice_metadata)} slices")
print(f"✓ Slices saved to: {SLICES_PATH}")
print(f"✓ Metadata saved to: {metadata_path}")

# Print statistics
labels = [m['label'] for m in slice_metadata]
print(f"\nSlice Statistics:")
print(f"Control (0): {labels.count(0)} slices")
print(f"Dementia (1): {labels.count(1)} slices")
