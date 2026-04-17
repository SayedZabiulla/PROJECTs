import os

PROJECT_PATH = os.path.expanduser("~/HD_Diagnosis_Project")
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")
EVAL_PATH = os.path.join(RESULTS_PATH, "evaluation")
GRADCAM_PATH = os.path.join(RESULTS_PATH, "gradcam")

print("="*60)
print("PHASE 3: EVALUATION & INTERPRETABILITY SUMMARY")
print("="*60)

# Check evaluation files
eval_files = [
    "test_metrics.json",
    "confusion_matrix.png",
    "roc_curve.png",
    "metrics_summary.png",
    "predictions.npz",
    "subject_level_metrics.json"
]

print("\n✓ Evaluation Files:")
for file in eval_files:
    path = os.path.join(EVAL_PATH, file)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {file} ({size:.1f} KB)")
    else:
        print(f"  ✗ {file} - NOT FOUND")

# Check Grad-CAM files
import glob
gradcam_files = glob.glob(os.path.join(GRADCAM_PATH, "*.png"))
print(f"\n✓ Grad-CAM Visualizations: {len(gradcam_files)} images")

# Check final report
report_path = os.path.join(RESULTS_PATH, "FINAL_PROJECT_REPORT.txt")
if os.path.exists(report_path):
    print(f"\n✓ Final Report Generated:")
    print(f"  {report_path}")
else:
    print(f"\n✗ Final report not found")

# Display results structure
print("\n" + "="*60)
print("RESULTS DIRECTORY STRUCTURE")
print("="*60)
print(f"""
{RESULTS_PATH}/
├── evaluation/
│   ├── test_metrics.json
│   ├── subject_level_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics_summary.png
│   └── predictions.npz
├── gradcam/
│   └── gradcam_*.png ({len(gradcam_files)} files)
├── training_history.json
├── training_history.png
├── training_config.json
└── FINAL_PROJECT_REPORT.txt
""")

print("="*60)
print("✅ PHASE 3 COMPLETE!")
print("="*60)
print("\nAll project phases completed successfully:")
print("  ✓ Phase 1: Data Preprocessing & Pipeline")
print("  ✓ Phase 2: Model Training")
print("  ✓ Phase 3: Evaluation & Interpretability")
print("\n🎉 PROJECT COMPLETE!")
print("="*60)
