import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd

# Paths
PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")
COMPARISON_PATH = os.path.join(RESULTS_PATH, "final_comparison")
os.makedirs(COMPARISON_PATH, exist_ok=True)

print("="*80)
print("3-WAY MODEL COMPARISON: ResNet50 vs DenseNet121 vs EfficientNet-B0")
print("="*80)

# Load all results
models_data = {}

for model_name, folder in [
    ('ResNet50', 'evaluation'),
    ('DenseNet121', 'densenet121/evaluation'),
    ('EfficientNet-B0', 'efficientnet_b0/evaluation')
]:
    eval_path = os.path.join(RESULTS_PATH, folder, "test_metrics.json")
    pred_path = os.path.join(RESULTS_PATH, folder, "predictions.npz")
    
    with open(eval_path, 'r') as f:
        metrics = json.load(f)
    
    predictions = np.load(pred_path)
    
    models_data[model_name] = {
        'metrics': metrics,
        'predictions': predictions
    }

# Also load training histories
for model_name, folder in [
    ('ResNet50', ''),
    ('DenseNet121', 'densenet121'),
    ('EfficientNet-B0', 'efficientnet_b0')
]:
    if folder:
        history_path = os.path.join(RESULTS_PATH, folder, "training_history.json")
    else:
        history_path = os.path.join(RESULTS_PATH, "training_history.json")
    
    with open(history_path, 'r') as f:
        models_data[model_name]['history'] = json.load(f)


def create_comparison_table():
    """Create comprehensive comparison table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    
    # Create comparison dictionary
    comparison = {}
    
    for model_name in ['ResNet50', 'DenseNet121', 'EfficientNet-B0']:
        metrics = models_data[model_name]['metrics']
        history = models_data[model_name]['history']
        
        comparison[model_name] = {
            'Architecture Year': '2015' if 'ResNet' in model_name else ('2017' if 'DenseNet' in model_name else '2019'),
            'Parameters': '24.5M' if 'ResNet' in model_name else ('7.5M' if 'DenseNet' in model_name else '5.3M'),
            'Test Accuracy': f"{metrics['accuracy']:.4f}",
            'Test Precision': f"{metrics['precision']:.4f}",
            'Test Recall': f"{metrics['recall']:.4f}",
            'Test Specificity': f"{metrics['specificity']:.4f}",
            'Test F1 Score': f"{metrics['f1_score']:.4f}",
            'Test ROC AUC': f"{metrics['roc_auc']:.4f}",
            'True Positives': metrics['true_positives'],
            'True Negatives': metrics['true_negatives'],
            'False Positives': metrics['false_positives'],
            'False Negatives': metrics['false_negatives'],
            'Best Val Accuracy': f"{max(history['val_acc']):.4f}",
            'Best Val F1': f"{max(history['val_f1']):.4f}",
            'Training Epochs': len(history['train_loss'])
        }
    
    # Print table
    print(f"\n{'Metric':<25} {'ResNet50':>18} {'DenseNet121':>18} {'EfficientNet-B0':>18}")
    print("-"*80)
    
    for key in comparison['ResNet50'].keys():
        print(f"{key:<25} {str(comparison['ResNet50'][key]):>18} "
              f"{str(comparison['DenseNet121'][key]):>18} "
              f"{str(comparison['EfficientNet-B0'][key]):>18}")
    
    print("="*80)
    
    # Save as JSON
    json_path = os.path.join(COMPARISON_PATH, "complete_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Comparison table saved to: {json_path}")
    
    return comparison


def plot_metrics_comparison():
    """Plot all metrics side by side"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'ROC AUC']
    
    resnet_values = [
        models_data['ResNet50']['metrics']['accuracy'],
        models_data['ResNet50']['metrics']['precision'],
        models_data['ResNet50']['metrics']['recall'],
        models_data['ResNet50']['metrics']['specificity'],
        models_data['ResNet50']['metrics']['f1_score'],
        models_data['ResNet50']['metrics']['roc_auc']
    ]
    
    densenet_values = [
        models_data['DenseNet121']['metrics']['accuracy'],
        models_data['DenseNet121']['metrics']['precision'],
        models_data['DenseNet121']['metrics']['recall'],
        models_data['DenseNet121']['metrics']['specificity'],
        models_data['DenseNet121']['metrics']['f1_score'],
        models_data['DenseNet121']['metrics']['roc_auc']
    ]
    
    efficientnet_values = [
        models_data['EfficientNet-B0']['metrics']['accuracy'],
        models_data['EfficientNet-B0']['metrics']['precision'],
        models_data['EfficientNet-B0']['metrics']['recall'],
        models_data['EfficientNet-B0']['metrics']['specificity'],
        models_data['EfficientNet-B0']['metrics']['f1_score'],
        models_data['EfficientNet-B0']['metrics']['roc_auc']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bars1 = ax.bar(x - width, resnet_values, width, label='ResNet50 (2015)', 
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, densenet_values, width, label='DenseNet121 (2017)', 
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, efficientnet_values, width, label='EfficientNet-B0 (2019)', 
                   color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('3-Way Architecture Comparison - Test Set Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(fontsize=13, loc='lower right', framealpha=0.95)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=2, label='Excellent (0.90)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.4, linewidth=2, label='Good (0.70)')
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "metrics_comparison_3way.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 3-way metrics comparison saved to: {save_path}")
    plt.close()


def plot_roc_comparison():
    """Plot all ROC curves together"""
    plt.figure(figsize=(12, 10))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for (model_name, color) in zip(['ResNet50', 'DenseNet121', 'EfficientNet-B0'], colors):
        pred_data = models_data[model_name]['predictions']
        fpr, tpr, _ = roc_curve(pred_data['y_true'], pred_data['y_scores'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=3, 
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier', alpha=0.6)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
    plt.title('ROC Curve Comparison - All Architectures', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=13, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "roc_comparison_3way.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 3-way ROC comparison saved to: {save_path}")
    plt.close()


def plot_confusion_matrices():
    """Plot all confusion matrices side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    models = ['ResNet50', 'DenseNet121', 'EfficientNet-B0']
    cmaps = ['Blues', 'Reds', 'Greens']
    
    for ax, model_name, cmap in zip(axes, models, cmaps):
        cm = np.array(models_data[model_name]['metrics']['confusion_matrix'])
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        annot = np.empty_like(cm, dtype=str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                    xticklabels=['Control', 'Dementia'],
                    yticklabels=['Control', 'Dementia'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 11}, linewidths=2, linecolor='black')
        
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=10)
    
    plt.suptitle('Confusion Matrices - All Architectures', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "confusion_matrices_3way.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 3-way confusion matrices saved to: {save_path}")
    plt.close()


def plot_efficiency_vs_performance():
    """Plot parameter efficiency vs performance"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Data
    params = [24.5, 7.5, 5.3]  # millions
    accuracy = [
        models_data['ResNet50']['metrics']['accuracy'],
        models_data['DenseNet121']['metrics']['accuracy'],
        models_data['EfficientNet-B0']['metrics']['accuracy']
    ]
    f1_scores = [
        models_data['ResNet50']['metrics']['f1_score'],
        models_data['DenseNet121']['metrics']['f1_score'],
        models_data['EfficientNet-B0']['metrics']['f1_score']
    ]
    models = ['ResNet50\n(2015)', 'DenseNet121\n(2017)', 'EfficientNet-B0\n(2019)']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Parameters comparison
    bars = axes[0].bar(models, params, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}M',
                     ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    axes[0].set_ylabel('Parameters (Millions)', fontsize=13, fontweight='bold')
    axes[0].set_title('Model Complexity', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, max(params) * 1.2])
    
    # Efficiency vs Performance scatter
    for i, (param, acc, f1, model, color) in enumerate(zip(params, accuracy, f1_scores, 
                                                             ['ResNet50', 'DenseNet121', 'EfficientNet-B0'], 
                                                             colors)):
        axes[1].scatter(param, acc, s=500, color=color, alpha=0.7, edgecolors='black', linewidth=2)
        axes[1].annotate(f'{model}\nAcc: {acc:.3f}\nF1: {f1:.3f}', 
                        xy=(param, acc), xytext=(10, -20 if i == 0 else 10),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
    
    axes[1].set_xlabel('Parameters (Millions)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_title('Efficiency vs Performance Trade-off', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.84, 0.90])
    
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "efficiency_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Efficiency analysis saved to: {save_path}")
    plt.close()


def plot_training_curves():
    """Compare training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'ResNet50': '#3498db', 'DenseNet121': '#e74c3c', 'EfficientNet-B0': '#2ecc71'}
    
    for model_name in ['ResNet50', 'DenseNet121', 'EfficientNet-B0']:
        history = models_data[model_name]['history']
        epochs = range(1, len(history['val_loss']) + 1)
        
        # Validation Loss
        axes[0, 0].plot(epochs, history['val_loss'], 
                       label=model_name, linewidth=2.5, marker='o', 
                       markersize=6, color=colors[model_name])
        
        # Validation Accuracy
        axes[0, 1].plot(epochs, history['val_acc'], 
                       label=model_name, linewidth=2.5, marker='s',
                       markersize=6, color=colors[model_name])
        
        # Validation F1
        axes[1, 0].plot(epochs, history['val_f1'], 
                       label=model_name, linewidth=2.5, marker='^',
                       markersize=6, color=colors[model_name])
        
        # Learning Rate
        axes[1, 1].plot(epochs, history['learning_rate'], 
                       label=model_name, linewidth=2.5, marker='d',
                       markersize=6, color=colors[model_name])
    
    # Configure subplots
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Validation F1 Score Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle('Training Dynamics - All Architectures', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    save_path = os.path.join(COMPARISON_PATH, "training_curves_3way.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves comparison saved to: {save_path}")
    plt.close()


def determine_winner():
    """Determine overall best model"""
    print("\n" + "="*80)
    print("🏆 WINNER ANALYSIS")
    print("="*80)
    
    scores = {}
    
    for model_name in ['ResNet50', 'DenseNet121', 'EfficientNet-B0']:
        metrics = models_data[model_name]['metrics']
        
        # Scoring system (normalized)
        score = (
            metrics['accuracy'] * 0.2 +
            metrics['recall'] * 0.3 +  # Most important for medical
            metrics['f1_score'] * 0.25 +
            metrics['roc_auc'] * 0.25
        )
        
        scores[model_name] = score
    
    winner = max(scores, key=scores.get)
    
    print(f"\n🥇 Overall Winner: {winner}")
    print(f"   Composite Score: {scores[winner]:.4f}")
    print(f"\n🥈 Second Place: {sorted(scores, key=scores.get, reverse=True)[1]}")
    print(f"🥉 Third Place: {sorted(scores, key=scores.get, reverse=True)[2]}")
    
    print(f"\n📊 Detailed Scores:")
    for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model}: {score:.4f}")
    
    # Clinical recommendation
    print(f"\n🏥 Clinical Recommendation:")
    
    best_recall_model = max(['ResNet50', 'DenseNet121', 'EfficientNet-B0'], 
                             key=lambda x: models_data[x]['metrics']['recall'])
    
    print(f"   Best for Screening: {best_recall_model} "
          f"(Recall: {models_data[best_recall_model]['metrics']['recall']:.2%})")
    
    most_efficient = 'EfficientNet-B0'
    print(f"   Most Efficient: {most_efficient} (5.3M parameters)")
    
    print("="*80)


# Run all comparisons
print("\n🎯 Generating comprehensive 3-way comparison...")
comparison_data = create_comparison_table()
plot_metrics_comparison()
plot_roc_comparison()
plot_confusion_matrices()
plot_efficiency_vs_performance()
plot_training_curves()
determine_winner()

print("\n" + "="*80)
print("✅ 3-WAY MODEL COMPARISON COMPLETE!")
print("="*80)
print(f"All results saved in: {COMPARISON_PATH}")
print("\n📁 Generated files:")
print("   - complete_comparison.json")
print("   - metrics_comparison_3way.png")
print("   - roc_comparison_3way.png")
print("   - confusion_matrices_3way.png")
print("   - efficiency_analysis.png")
print("   - training_curves_3way.png")
print("="*80)
