import os
import numpy as np
import pickle
import json
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

from model import create_model
from create_dataset import MRISliceDataset, val_test_transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data/processed")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")
EVAL_PATH = os.path.join(RESULTS_PATH, "evaluation")

MODEL_TO_USE = 'best_model_f1.pth'

print("="*60)
print("SUBJECT-LEVEL EVALUATION")
print("="*60)
print("Aggregating slice-level predictions to subject-level")
print("="*60)


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load test data
    splits_path = os.path.join(PROCESSED_DATA_PATH, "data_splits.pkl")
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    test_data = splits['test']
    
    # Load model
    model = create_model(pretrained=False, freeze_backbone=False, device=device)
    model_path = os.path.join(MODELS_PATH, MODEL_TO_USE)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded")
    
    # Group slices by subject
    subject_slices = defaultdict(list)
    for sample in test_data:
        subject_id = sample['subject_id']
        subject_slices[subject_id].append(sample)
    
    print(f"\n✓ Total test subjects: {len(subject_slices)}")
    
    # Evaluate each subject
    subject_results = []
    
    print("\nEvaluating subjects...")
    for subject_id, slices in subject_slices.items():
        # Get true label (same for all slices of a subject)
        true_label = slices[0]['label']
        
        # Get predictions for all slices
        slice_probs = []
        
        with torch.no_grad():
            for slice_data in slices:
                # Load and preprocess
                image = np.load(slice_data['slice_path'])
                image = np.squeeze(image)
                image_tensor = np.expand_dims(image, axis=0).astype(np.float32)
                image_tensor = val_test_transforms(image_tensor)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                # Predict
                output = model(image_tensor).squeeze()
                prob = torch.sigmoid(output).item()
                slice_probs.append(prob)
        
        # Aggregate predictions (majority voting and average probability)
        avg_prob = np.mean(slice_probs)
        majority_vote = 1 if avg_prob > 0.5 else 0
        
        subject_results.append({
            'subject_id': subject_id,
            'true_label': true_label,
            'predicted_label': majority_vote,
            'avg_probability': avg_prob,
            'num_slices': len(slices),
            'slice_probabilities': slice_probs
        })
    
    # Calculate subject-level metrics
    y_true = [r['true_label'] for r in subject_results]
    y_pred = [r['predicted_label'] for r in subject_results]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print("SUBJECT-LEVEL RESULTS")
    print("="*60)
    print(f"\n📊 Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")
    
    # Save results
    subject_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'num_subjects': len(subject_results),
        'subject_results': [
            {
                'subject_id': r['subject_id'],
                'true_label': r['true_label'],
                'predicted_label': r['predicted_label'],
                'avg_probability': float(r['avg_probability']),
                'num_slices': r['num_slices']
            }
            for r in subject_results
        ]
    }
    
    save_path = os.path.join(EVAL_PATH, "subject_level_metrics.json")
    with open(save_path, 'w') as f:
        json.dump(subject_metrics, f, indent=2)
    
    print(f"\n✓ Subject-level metrics saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
