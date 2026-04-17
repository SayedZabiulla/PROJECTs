import torch
import monai
import nibabel as nib
import pandas as pd

print(f"Python: {__import__('sys').version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"MONAI: {monai.__version__}")
print(f"NiBabel: {nib.__version__}")
print(f"Pandas: {pd.__version__}")
