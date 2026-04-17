import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

from model_efficientnet import create_efficientnet_model
from create_dataset import MRISliceDataset, val_test_transforms
from torch.utils.data import DataLoader

# Paths
PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data/processed")
MODELS_PATH = os.path.join(PROJECT_PATH, "models/efficientnet_b0")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results/efficientnet_b0")

# Create evaluation results directory
EVAL_PATH = os.path.join(RESULTS_PATH, "evaluation")
os.makedirs(EVAL_PATH, exist_ok=True)

# Configuration
BATCH_SIZE = 16
NUM_WORKERS = 4
MODEL_TO_EVALUATE = 'best_model_f1.pth'

print("="*60)
print("EFFICIENTNET-B0 EVALUATION ON TEST SET")
print("="*60)
print(f"Model: {MODEL_TO_EVALUATE}")
print("="*60)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def calculate_detailed_metrics(y_true, y_pred, y_scores):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    return metrics


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.empty_like(cm, dtype=str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens', 
                xticklabels=['Control (0)', 'Dementia (1)'],
                yticklabels=['Control (0)', 'Dementia (1)'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('EfficientNet-B0 - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='green', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('EfficientNet-B0 - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {save_path}")
    plt.close()


def plot_metrics_bar(metrics, save_path):
    """Plot metrics as bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'Specificity', 'F1 Score', 'ROC AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('EfficientNet-B0 - Performance Metrics (Test Set)', fontsize=14, fontweight='bold')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.90)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.70)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics bar chart saved to: {save_path}")
    plt.close()


def print_evaluation_report(metrics, y_true, y_pred):
    """Print detailed evaluation report"""
    print("\n" + "="*60)
    print("EFFICIENTNET-B0 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Classification Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    
    print(f"\n📈 Confusion Matrix Breakdown:")
    print(f"  True Positives (TP):  {metrics['true_positives']}")
    print(f"  True Negatives (TN):  {metrics['true_negatives']}")
    print(f"  False Positives (FP): {metrics['false_positives']}")
    print(f"  False Negatives (FN): {metrics['false_negatives']}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Control', 'Dementia'],
                                digits=4))
    
    print("="*60)


def save_metrics_json(metrics, save_path):
    """Save metrics to JSON file"""
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics['specificity']),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'true_positives': int(metrics['true_positives']),
        'true_negatives': int(metrics['true_negatives']),
        'false_positives': int(metrics['false_positives']),
        'false_negatives': int(metrics['false_negatives'])
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"✓ Metrics saved to: {save_path}")


def main():
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
    print(f"Test samples: {len(test_data)}")
    
    test_labels = [d['label'] for d in test_data]
    print(f"  Control (0): {test_labels.count(0)}")
    print(f"  Dementia (1): {test_labels.count(1)}")
    
    test_dataset = MRISliceDataset(test_data, transform=val_test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✓ Test DataLoader created ({len(test_loader)} batches)")
    
    # Load model
    print("\n" + "="*60)
    print("LOADING EFFICIENTNET-B0 MODEL")
    print("="*60)
    
    model = create_efficientnet_model(pretrained=False, freeze_backbone=False, device=device)
    
    model_path = os.path.join(MODELS_PATH, MODEL_TO_EVALUATE)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Trained for {checkpoint['epoch']} epochs")
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    y_true, y_pred, y_scores = evaluate_model(model, test_loader, device)
    metrics = calculate_detailed_metrics(y_true, y_pred, y_scores)
    
    print_evaluation_report(metrics, y_true, y_pred)
    
    # Save metrics
    metrics_path = os.path.join(EVAL_PATH, "test_metrics.json")
    save_metrics_json(metrics, metrics_path)
    
    # Plot visualizations
    cm_path = os.path.join(EVAL_PATH, "confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    
    roc_path = os.path.join(EVAL_PATH, "roc_curve.png")
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], roc_path)
    
    metrics_bar_path = os.path.join(EVAL_PATH, "metrics_summary.png")
    plot_metrics_bar(metrics, metrics_bar_path)
    
    # Save predictions
    predictions_path = os.path.join(EVAL_PATH, "predictions.npz")
    np.savez(predictions_path, 
             y_true=y_true, 
             y_pred=y_pred, 
             y_scores=y_scores)
    print(f"✓ Predictions saved to: {predictions_path}")
    
    print("\n" + "="*60)
    print("EFFICIENTNET-B0 EVALUATION COMPLETE!")
    print("="*60)
    print(f"Results saved in: {EVAL_PATH}")


if __name__ == "__main__":
    main()
