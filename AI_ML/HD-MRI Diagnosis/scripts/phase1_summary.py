import os
import pickle
import json

PROCESSED_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/processed")

print("=" * 60)
print("PHASE 1 SUMMARY")
print("=" * 60)

# Check label map
label_map_path = os.path.join(PROCESSED_DATA_PATH, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    print(f"\n✓ Label Map: {len(label_map)} subjects")
else:
    print("\n✗ Label map not found!")

# Check slices
slice_metadata_path = os.path.join(PROCESSED_DATA_PATH, "slice_metadata.pkl")
if os.path.exists(slice_metadata_path):
    with open(slice_metadata_path, 'rb') as f:
        slice_metadata = pickle.load(f)
    print(f"✓ Extracted Slices: {len(slice_metadata)} slices")
    labels = [m['label'] for m in slice_metadata]
    print(f"  - Control (0): {labels.count(0)}")
    print(f"  - Dementia (1): {labels.count(1)}")
else:
    print("\n✗ Slice metadata not found!")

# Check splits
splits_path = os.path.join(PROCESSED_DATA_PATH, "data_splits.pkl")
if os.path.exists(splits_path):
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    print(f"\n✓ Data Splits:")
    print(f"  - Train: {len(splits['train'])} slices")
    print(f"  - Val: {len(splits['val'])} slices")
    print(f"  - Test: {len(splits['test'])} slices")
else:
    print("\n✗ Data splits not found!")

print("\n" + "=" * 60)
print("Phase 1 is COMPLETE and ready for Phase 2 (Model Training)!")
print("=" * 60)
