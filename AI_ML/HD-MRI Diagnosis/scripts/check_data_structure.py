import os
import glob

# Set paths
RAW_DATA_PATH = os.path.expanduser("~/HD_Diagnosis_Project/data/raw")
CSV_PATH = os.path.join(RAW_DATA_PATH, "oasis_cross-sectional.csv")

# Check CSV exists
if os.path.exists(CSV_PATH):
    print(f"✓ CSV file found: {CSV_PATH}")
else:
    print(f"✗ CSV file NOT found: {CSV_PATH}")

# Search for all disc folders (disc1 to disc12)
all_subject_folders = []

for disc_num in range(1, 13):  # disc1 to disc12
    disc_path = os.path.join(RAW_DATA_PATH, f"oasis_cross-sectional_disc{disc_num}/disc{disc_num}")
    
    if os.path.exists(disc_path):
        subject_folders = glob.glob(os.path.join(disc_path, "OAS1_*_MR*"))
        all_subject_folders.extend(subject_folders)
        print(f"✓ Disc {disc_num}: Found {len(subject_folders)} subjects")
    else:
        print(f"✗ Disc {disc_num}: Path not found - {disc_path}")

# Sort all subjects
all_subject_folders = sorted(all_subject_folders)

print(f"\n{'='*60}")
print(f"TOTAL SUBJECTS FOUND: {len(all_subject_folders)}")
print(f"{'='*60}")
print(f"First 5 subjects: {[os.path.basename(f) for f in all_subject_folders[:5]]}")
print(f"Last 5 subjects: {[os.path.basename(f) for f in all_subject_folders[-5:]]}")

# Check scan files in first subject's RAW folder
if all_subject_folders:
    first_subject = all_subject_folders[0]
    raw_folder = os.path.join(first_subject, "RAW")
    
    print(f"\n{'='*60}")
    print(f"SAMPLE SUBJECT ANALYSIS")
    print(f"{'='*60}")
    print(f"Subject: {os.path.basename(first_subject)}")
    print(f"RAW folder: {raw_folder}")
    print(f"RAW folder exists: {os.path.exists(raw_folder)}")
    
    # Check for different file types
    img_files = glob.glob(os.path.join(raw_folder, "*.img"))
    hdr_files = glob.glob(os.path.join(raw_folder, "*.hdr"))
    gif_files = glob.glob(os.path.join(raw_folder, "*.gif"))
    
    print(f"\nFiles in RAW folder:")
    print(f"  - .img files: {len(img_files)}")
    print(f"  - .hdr files: {len(hdr_files)}")
    print(f"  - .gif files: {len(gif_files)}")
    
    if img_files:
        print(f"\nExample files:")
        for i, img_file in enumerate(img_files[:3], 1):
            print(f"  {i}. {os.path.basename(img_file)}")

# Optional: Check distribution of subjects across discs
print(f"\n{'='*60}")
print(f"DISTRIBUTION SUMMARY")
print(f"{'='*60}")
for disc_num in range(1, 13):
    disc_path = os.path.join(RAW_DATA_PATH, f"oasis_cross-sectional_disc{disc_num}/disc{disc_num}")
    if os.path.exists(disc_path):
        count = len(glob.glob(os.path.join(disc_path, "OAS1_*_MR*")))
        print(f"Disc {disc_num:2d}: {count:3d} subjects")
