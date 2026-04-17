import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Binary(nn.Module):
    """DenseNet121 adapted for binary classification of MRI slices"""
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(DenseNet121Binary, self).__init__()
        
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modify first conv layer to accept 1-channel input (grayscale MRI)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Copy pretrained weights for the first channel (if pretrained)
        if pretrained:
            with torch.no_grad():
                # Average the RGB weights to create single-channel weights
                self.densenet.features.conv0.weight[:, 0, :, :] = \
                    original_conv.weight.mean(dim=1)
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.densenet.features.parameters():
                param.requires_grad = False
        
        # Replace final classifier
        # Original: Linear(in_features=1024, out_features=1000)
        # New: Custom classifier for binary classification
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)  # Binary output
        )
    
    def forward(self, x):
        return self.densenet(x)


def create_densenet_model(pretrained=True, freeze_backbone=False, device='cuda'):
    """
    Create and initialize the DenseNet121 model
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze DenseNet backbone layers
        device: Device to load model on
    
    Returns:
        model: DenseNet121Binary model
    """
    model = DenseNet121Binary(pretrained=pretrained, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Print model summary
    print("="*60)
    print("DENSENET121 MODEL ARCHITECTURE")
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
    model = create_densenet_model(pretrained=True, freeze_backbone=False, device=device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output.squeeze()}")
