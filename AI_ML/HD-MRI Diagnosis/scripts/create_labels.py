import pandas as pd
import os
import json

# Paths
RAW_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/raw")
PROCESSED_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/processed")
CSV_PATH = os.path.join(RAW_DATA_PATH, "oasis_cross-sectional.csv")

# Load clinical data
print("Loading clinical data...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Check CDR distribution
print(f"\nCDR Distribution:\n{df['CDR'].value_counts().sort_index()}")

# Create binary labels
# 0 = Control (CDR == 0.0)
# 1 = Dementia (CDR > 0.0)
df['Label'] = (df['CDR'] > 0.0).astype(int)

# Create label map: ID -> Label
label_map = {}
for _, row in df.iterrows():
    subject_id = row['ID']
    label = row['Label']
    label_map[subject_id] = label

# Print label distribution
print(f"\nLabel Distribution:")
print(f"Control (0): {sum(1 for v in label_map.values() if v == 0)}")
print(f"Dementia (1): {sum(1 for v in label_map.values() if v == 1)}")

# Save label map
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
label_map_path = os.path.join(PROCESSED_DATA_PATH, "label_map.json")
with open(label_map_path, 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"\n✓ Label map saved to: {label_map_path}")

# Save processed dataframe
df_processed_path = os.path.join(PROCESSED_DATA_PATH, "clinical_data_with_labels.csv")
df.to_csv(df_processed_path, index=False)
print(f"✓ Processed clinical data saved to: {df_processed_path}")
