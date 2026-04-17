import torch
import torch.nn as nn
from torchvision import models

class ResNet50Binary(nn.Module):
    """ResNet50 adapted for binary classification of MRI slices"""
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(ResNet50Binary, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer to accept 1-channel input (grayscale MRI)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer
        # Original: Linear(in_features=2048, out_features=1000)
        # New: Linear(in_features=2048, out_features=1) for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)  # Binary output
        )
    
    def forward(self, x):
        return self.resnet(x)


def create_model(pretrained=True, freeze_backbone=False, device='cuda'):
    """
    Create and initialize the model
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze ResNet backbone layers
        device: Device to load model on
    
    Returns:
        model: ResNet50Binary model
    """
    model = ResNet50Binary(pretrained=pretrained, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Print model summary
    print("="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"Pretrained: {pretrained}")
    print(f"Backbone frozen: {freeze_backbone}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(pretrained=True, freeze_backbone=False, device=device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output.squeeze()}")
