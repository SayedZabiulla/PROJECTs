import os
import glob
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm
import pickle

# Paths
RAW_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/raw")
PROCESSED_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/processed")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_PATH, "label_map.json")

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

# Get all subject folders from all 12 discs
all_subject_folders = []

print("Searching for subjects across all discs...")
for disc_num in range(1, 13):  # disc1 to disc12
    disc_path = os.path.join(RAW_DATA_PATH, f"oasis_cross-sectional_disc{disc_num}/disc{disc_num}")
    
    if os.path.exists(disc_path):
        subject_folders = glob.glob(os.path.join(disc_path, "OAS1_*_MR*"))
        all_subject_folders.extend(subject_folders)
        print(f"  Disc {disc_num}: Found {len(subject_folders)} subjects")
    else:
        print(f"  Disc {disc_num}: Path not found")

# Sort all subjects
all_subject_folders = sorted(all_subject_folders)
print(f"\n✓ Total subjects found: {len(all_subject_folders)}")

# Create metadata list
metadata = []

print("\nProcessing subjects...")
for subject_folder in tqdm(all_subject_folders):
    subject_id = os.path.basename(subject_folder)
    
    # Check if subject has label
    if subject_id not in label_map:
        print(f"Warning: {subject_id} not found in label map, skipping...")
        continue
    
    label = label_map[subject_id]
    
    # Find all .img files in the RAW folder (updated path)
    raw_folder = os.path.join(subject_folder, "RAW")
    img_files = glob.glob(os.path.join(raw_folder, "*mpr*.img"))
    
    if len(img_files) == 0:
        print(f"Warning: No .img files found for {subject_id}, skipping...")
        continue
    
    # Load all scans
    scan_data = []
    for img_file in img_files:
        try:
            # Load using nibabel (Analyze format)
            img = nib.load(img_file)
            data = img.get_fdata()
            scan_data.append(data)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    if len(scan_data) == 0:
        print(f"Warning: Could not load any scans for {subject_id}, skipping...")
        continue
    
    # Average scans
    averaged_volume = np.mean(scan_data, axis=0).astype(np.float32)
    
    # Save metadata
    metadata.append({
        'subject_id': subject_id,
        'label': label,
        'num_scans_averaged': len(scan_data),
        'volume_shape': averaged_volume.shape,
        'volume_path': None  # We'll store slices, not full volumes
    })
    
    # Optional: Save averaged volume (only if you have disk space)
    # volume_save_path = os.path.join(PROCESSED_DATA_PATH, 'volumes', f'{subject_id}.npy')
    # os.makedirs(os.path.dirname(volume_save_path), exist_ok=True)
    # np.save(volume_save_path, averaged_volume)

# Save metadata
metadata_path = os.path.join(PROCESSED_DATA_PATH, "volume_metadata.pkl")
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n✓ Processed {len(metadata)} subjects")
print(f"✓ Metadata saved to: {metadata_path}")

# Print statistics
labels = [m['label'] for m in metadata]
print(f"\nFinal Dataset Statistics:")
print(f"Control (0): {labels.count(0)}")
print(f"Dementia (1): {labels.count(1)}")
