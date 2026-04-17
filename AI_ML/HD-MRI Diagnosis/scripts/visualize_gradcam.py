import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
import random

from model import create_model
from create_dataset import MRISliceDataset, val_test_transforms
from gradcam_utils import GradCAM, visualize_gradcam

# Paths
PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data/processed")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")

# Create Grad-CAM results directory
GRADCAM_PATH = os.path.join(RESULTS_PATH, "gradcam")
os.makedirs(GRADCAM_PATH, exist_ok=True)

# Configuration
MODEL_TO_USE = 'best_model_f1.pth'
NUM_SAMPLES = 20  # Number of samples to visualize
RANDOM_SEED = 42

print("="*60)
print("GRAD-CAM VISUALIZATION")
print("="*60)
print(f"Model: {MODEL_TO_USE}")
print(f"Samples to visualize: {NUM_SAMPLES}")
print("="*60)


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load test data
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    splits_path = os.path.join(PROCESSED_DATA_PATH, "data_splits.pkl")
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    test_data = splits['test']
    print(f"✓ Test samples: {len(test_data)}")
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    model = create_model(pretrained=False, freeze_backbone=False, device=device)
    
    model_path = os.path.join(MODELS_PATH, MODEL_TO_USE)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {model_path}")
    
    # Get target layer (last conv layer of ResNet50)
    target_layer = model.resnet.layer4[-1].conv3
    print(f"✓ Target layer: resnet.layer4[-1].conv3")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    print(f"✓ Grad-CAM initialized")
    
    # Select random samples
    random.seed(RANDOM_SEED)
    
    # Get samples from each class
    control_samples = [s for s in test_data if s['label'] == 0]
    dementia_samples = [s for s in test_data if s['label'] == 1]
    
    num_per_class = NUM_SAMPLES // 2
    selected_control = random.sample(control_samples, min(num_per_class, len(control_samples)))
    selected_dementia = random.sample(dementia_samples, min(num_per_class, len(dementia_samples)))
    
    selected_samples = selected_control + selected_dementia
    random.shuffle(selected_samples)
    
    print(f"\n✓ Selected {len(selected_samples)} samples:")
    print(f"  Control: {len(selected_control)}")
    print(f"  Dementia: {len(selected_dementia)}")
    
    # Generate Grad-CAM visualizations
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*60)
    
    correct_count = 0
    
    for idx, sample in enumerate(selected_samples):
        print(f"\nProcessing sample {idx+1}/{len(selected_samples)}...")
        
        # Load image
        slice_path = sample['slice_path']
        true_label = sample['label']
        subject_id = sample['subject_id']
        slice_idx = sample['slice_idx']
        
        # Load and preprocess
        image = np.load(slice_path)
        image = np.squeeze(image)
        
        # Apply transforms
        image_tensor = np.expand_dims(image, axis=0).astype(np.float32)
        image_tensor = val_test_transforms(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 1, 224, 224)
        
        # Generate Grad-CAM
        heatmap, prediction, prob = grad_cam.generate_cam(image_tensor)
        
        # Track accuracy
        if prediction == true_label:
            correct_count += 1
        
        # Get original image for visualization (after resize but before normalization)
        from monai.transforms import Resize
        resize_transform = Resize(spatial_size=(224, 224))
        image_resized = resize_transform(np.expand_dims(image, axis=0))[0]
        
        # Create visualization
        save_name = f"gradcam_{idx+1:02d}_{subject_id}_slice{slice_idx:03d}_true{true_label}_pred{int(prediction)}.png"
        save_path = os.path.join(GRADCAM_PATH, save_name)
        
        visualize_gradcam(
            original_image=image_resized,
            heatmap=heatmap,
            prediction=int(prediction),
            true_label=true_label,
            prob=prob,
            save_path=save_path
        )
        
        print(f"  ✓ Saved: {save_name}")
        print(f"    True: {true_label}, Pred: {int(prediction)}, Prob: {prob:.2%}")
    
    # Summary
    accuracy = correct_count / len(selected_samples)
    
    print("\n" + "="*60)
    print("GRAD-CAM VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"Total samples visualized: {len(selected_samples)}")
    print(f"Correct predictions: {correct_count}/{len(selected_samples)} ({accuracy:.2%})")
    print(f"Results saved in: {GRADCAM_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
