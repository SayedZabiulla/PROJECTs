import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Paths
PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")
COMPARISON_PATH = os.path.join(RESULTS_PATH, "model_comparison")
os.makedirs(COMPARISON_PATH, exist_ok=True)

# Load ResNet50 results
resnet_eval_path = os.path.join(RESULTS_PATH, "evaluation/test_metrics.json")
resnet_train_path = os.path.join(RESULTS_PATH, "training_history.json")

# Load DenseNet121 results
densenet_eval_path = os.path.join(RESULTS_PATH, "densenet121/evaluation/test_metrics.json")
densenet_train_path = os.path.join(RESULTS_PATH, "densenet121/training_history.json")

with open(resnet_eval_path, 'r') as f:
    resnet_metrics = json.load(f)

with open(resnet_train_path, 'r') as f:
    resnet_history = json.load(f)

with open(densenet_eval_path, 'r') as f:
    densenet_metrics = json.load(f)

with open(densenet_train_path, 'r') as f:
    densenet_history = json.load(f)

print("="*70)
print("MODEL COMPARISON: ResNet50 vs DenseNet121")
print("="*70)


def plot_comparison_bar():
    """Compare test metrics side by side"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'ROC AUC']
    
    resnet_values = [
        resnet_metrics['accuracy'],
        resnet_metrics['precision'],
        resnet_metrics['recall'],
        resnet_metrics['specificity'],
        resnet_metrics['f1_score'],
        resnet_metrics['roc_auc']
    ]
    
    densenet_values = [
        densenet_metrics['accuracy'],
        densenet_metrics['precision'],
        densenet_metrics['recall'],
        densenet_metrics['specificity'],
        densenet_metrics['f1_score'],
        densenet_metrics['roc_auc']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width/2, resnet_values, width, label='ResNet50', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, densenet_values, width, label='DenseNet121', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison - Test Set Performance', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    plt.close()


def plot_training_comparison():
    """Compare training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs_resnet = range(1, len(resnet_history['train_loss']) + 1)
    epochs_densenet = range(1, len(densenet_history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs_resnet, resnet_history['val_loss'], 
                    label='ResNet50', marker='o', linewidth=2, color='#3498db')
    axes[0, 0].plot(epochs_densenet, densenet_history['val_loss'], 
                    label='DenseNet121', marker='s', linewidth=2, color='#e74c3c')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Validation Loss', fontsize=11)
    axes[0, 0].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs_resnet, resnet_history['val_acc'], 
                    label='ResNet50', marker='o', linewidth=2, color='#3498db')
    axes[0, 1].plot(epochs_densenet, densenet_history['val_acc'], 
                    label='DenseNet121', marker='s', linewidth=2, color='#e74c3c')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Validation Accuracy', fontsize=11)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs_resnet, resnet_history['val_f1'], 
                    label='ResNet50', marker='o', linewidth=2, color='#3498db')
    axes[1, 0].plot(epochs_densenet, densenet_history['val_f1'], 
                    label='DenseNet121', marker='s', linewidth=2, color='#e74c3c')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Validation F1 Score', fontsize=11)
    axes[1, 0].set_title('Validation F1 Score Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Model Complexity
    model_names = ['ResNet50', 'DenseNet121']
    params = [24.5, 7.5]  # in millions
    colors = ['#3498db', '#e74c3c']
    
    bars = axes[1, 1].bar(model_names, params, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}M',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[1, 1].set_ylabel('Parameters (Millions)', fontsize=11)
    axes[1, 1].set_title('Model Complexity', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "training_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training comparison saved to: {save_path}")
    plt.close()


def plot_confusion_matrices():
    """Plot confusion matrices side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    resnet_cm = np.array(resnet_metrics['confusion_matrix'])
    densenet_cm = np.array(densenet_metrics['confusion_matrix'])
    
    for ax, cm, title, cmap in zip(axes, 
                                     [resnet_cm, densenet_cm], 
                                     ['ResNet50', 'DenseNet121'],
                                     ['Blues', 'Reds']):
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        annot = np.empty_like(cm, dtype=str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                    xticklabels=['Control', 'Dementia'],
                    yticklabels=['Control', 'Dementia'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 12})
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'{title} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "confusion_matrices_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices comparison saved to: {save_path}")
    plt.close()


def plot_roc_comparison():
    """Plot ROC curves comparison"""
    # Load predictions
    resnet_pred_path = os.path.join(RESULTS_PATH, "evaluation/predictions.npz")
    densenet_pred_path = os.path.join(RESULTS_PATH, "densenet121/evaluation/predictions.npz")
    
    resnet_pred = np.load(resnet_pred_path)
    densenet_pred = np.load(densenet_pred_path)
    
    from sklearn.metrics import roc_curve, auc
    
    resnet_fpr, resnet_tpr, _ = roc_curve(resnet_pred['y_true'], resnet_pred['y_scores'])
    densenet_fpr, densenet_tpr, _ = roc_curve(densenet_pred['y_true'], densenet_pred['y_scores'])
    
    resnet_auc = auc(resnet_fpr, resnet_tpr)
    densenet_auc = auc(densenet_fpr, densenet_tpr)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(resnet_fpr, resnet_tpr, color='#3498db', lw=3, 
             label=f'ResNet50 (AUC = {resnet_auc:.4f})')
    plt.plot(densenet_fpr, densenet_tpr, color='#e74c3c', lw=3, 
             label=f'DenseNet121 (AUC = {densenet_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('ROC Curve Comparison', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "roc_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC comparison saved to: {save_path}")
    plt.close()


def create_comparison_table():
    """Create detailed comparison table"""
    comparison_data = {
        'ResNet50': {
            'Parameters': '24.5M',
            'Test Accuracy': f"{resnet_metrics['accuracy']:.4f}",
            'Test Precision': f"{resnet_metrics['precision']:.4f}",
            'Test Recall': f"{resnet_metrics['recall']:.4f}",
            'Test Specificity': f"{resnet_metrics['specificity']:.4f}",
            'Test F1 Score': f"{resnet_metrics['f1_score']:.4f}",
            'Test ROC AUC': f"{resnet_metrics['roc_auc']:.4f}",
            'Best Val Accuracy': f"{max(resnet_history['val_acc']):.4f}",
            'Best Val F1': f"{max(resnet_history['val_f1']):.4f}",
            'Training Epochs': len(resnet_history['train_loss'])
        },
        'DenseNet121': {
            'Parameters': '7.5M',
            'Test Accuracy': f"{densenet_metrics['accuracy']:.4f}",
            'Test Precision': f"{densenet_metrics['precision']:.4f}",
            'Test Recall': f"{densenet_metrics['recall']:.4f}",
            'Test Specificity': f"{densenet_metrics['specificity']:.4f}",
            'Test F1 Score': f"{densenet_metrics['f1_score']:.4f}",
            'Test ROC AUC': f"{densenet_metrics['roc_auc']:.4f}",
            'Best Val Accuracy': f"{max(densenet_history['val_acc']):.4f}",
            'Best Val F1': f"{max(densenet_history['val_f1']):.4f}",
            'Training Epochs': len(densenet_history['train_loss'])
        }
    }
    
    # Save as JSON
    json_path = os.path.join(COMPARISON_PATH, "comparison_table.json")
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✓ Comparison table saved to: {json_path}")
    
    # Print table
    print("\n" + "="*70)
    print("DETAILED COMPARISON TABLE")
    print("="*70)
    print(f"{'Metric':<25} {'ResNet50':>15} {'DenseNet121':>15} {'Winner':>10}")
    print("-"*70)
    
    for key in comparison_data['ResNet50'].keys():
        resnet_val = comparison_data['ResNet50'][key]
        densenet_val = comparison_data['DenseNet121'][key]
        
        # Determine winner (for numeric values)
        if key != 'Parameters' and key != 'Training Epochs':
            try:
                winner = 'DenseNet' if float(densenet_val) > float(resnet_val) else 'ResNet'
            except:
                winner = '-'
        elif key == 'Parameters':
            winner = 'DenseNet'  # Fewer is better
        else:
            winner = '-'
        
        print(f"{key:<25} {resnet_val:>15} {densenet_val:>15} {winner:>10}")
    
    print("="*70)


# Run all comparisons
print("\n🎯 Generating comparison visualizations...")
plot_comparison_bar()
plot_training_comparison()
plot_confusion_matrices()
plot_roc_comparison()
create_comparison_table()

print("\n✅ MODEL COMPARISON COMPLETE!")
print(f"Results saved in: {COMPARISON_PATH}")
