import os
import shutil

PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")

print("="*60)
print("PREPARING PROJECT FOR GITHUB")
print("="*60)

# Create GitHub directory structure
github_path = os.path.join(PROJECT_PATH, "github_upload")
os.makedirs(github_path, exist_ok=True)

# Directories to copy
dirs_to_copy = [
    "scripts",
    "results/evaluation",
    "results/gradcam"
]

# Copy scripts
print("\n✓ Copying project files...")
for dir_name in dirs_to_copy:
    src = os.path.join(PROJECT_PATH, dir_name)
    if "results" in dir_name:
        dst = os.path.join(github_path, dir_name)
    else:
        dst = os.path.join(github_path, dir_name)
    
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  ✓ Copied {dir_name}")

# Copy important files
files_to_copy = [
    "results/training_history.json",
    "results/training_history.png",
    "results/training_config.json",
    "results/FINAL_PROJECT_REPORT.txt"
]

for file_name in files_to_copy:
    src = os.path.join(PROJECT_PATH, file_name)
    dst = os.path.join(github_path, file_name)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  ✓ Copied {file_name}")

print(f"\n✓ GitHub upload directory created at:")
print(f"  {github_path}")
print("="*60)
