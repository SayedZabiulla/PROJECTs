import os
import json
import torch

PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")

print("="*60)
print("PHASE 2: TRAINING SUMMARY")
print("="*60)

# Check saved models
models_to_check = [
    'best_model_loss.pth',
    'best_model_f1.pth',
    'final_model.pth'
]

print("\n✓ Saved Models:")
for model_name in models_to_check:
    model_path = os.path.join(MODELS_PATH, model_name)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  - {model_name}: {size_mb:.2f} MB")
    else:
        print(f"  ✗ {model_name}: Not found")

# Load training history
history_path = os.path.join(RESULTS_PATH, 'training_history.json')
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"\n✓ Training History:")
    print(f"  Total Epochs: {len(history['train_loss'])}")
    print(f"  Best Train Loss: {min(history['train_loss']):.4f}")
    print(f"  Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"  Best Train Accuracy: {max(history['train_acc']):.4f}")
    print(f"  Best Val Accuracy: {max(history['val_acc']):.4f}")
    print(f"  Best Train F1: {max(history['train_f1']):.4f}")
    print(f"  Best Val F1: {max(history['val_f1']):.4f}")
    
    # Find best epoch
    best_epoch = history['val_f1'].index(max(history['val_f1'])) + 1
    print(f"\n✓ Best Epoch: {best_epoch}")
    print(f"  Val Loss: {history['val_loss'][best_epoch-1]:.4f}")
    print(f"  Val Accuracy: {history['val_acc'][best_epoch-1]:.4f}")
    print(f"  Val F1 Score: {history['val_f1'][best_epoch-1]:.4f}")
else:
    print("\n✗ Training history not found")

# Check results
results_files = [
    'training_history.png',
    'training_config.json'
]

print("\n✓ Result Files:")
for result_file in results_files:
    result_path = os.path.join(RESULTS_PATH, result_file)
    if os.path.exists(result_path):
        print(f"  - {result_file}")
    else:
        print(f"  ✗ {result_file}: Not found")

print("\n" + "="*60)
print("Phase 2 Complete! Ready for Phase 3 (Model Evaluation)")
print("="*60)
