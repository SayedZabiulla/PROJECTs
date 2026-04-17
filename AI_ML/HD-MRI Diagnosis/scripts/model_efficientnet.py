import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB0Binary(nn.Module):
    """EfficientNet-B0 adapted for binary classification of MRI slices"""
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(EfficientNetB0Binary, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)
        
        # Modify first conv layer to accept 1-channel input (grayscale MRI)
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # New: Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        original_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        # Copy pretrained weights for the first channel (if pretrained)
        if pretrained:
            with torch.no_grad():
                # Average the RGB weights to create single-channel weights
                self.efficientnet.features[0][0].weight[:, 0, :, :] = \
                    original_conv.weight.mean(dim=1)
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False
        
        # Replace final classifier
        # Original: Linear(in_features=1280, out_features=1000)
        # New: Custom classifier for binary classification
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)  # Binary output
        )
    
    def forward(self, x):
        return self.efficientnet(x)


def create_efficientnet_model(pretrained=True, freeze_backbone=False, device='cuda'):
    """
    Create and initialize the EfficientNet-B0 model
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze EfficientNet backbone layers
        device: Device to load model on
    
    Returns:
        model: EfficientNetB0Binary model
    """
    model = EfficientNetB0Binary(pretrained=pretrained, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Print model summary
    print("="*60)
    print("EFFICIENTNET-B0 MODEL ARCHITECTURE")
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
    model = create_efficientnet_model(pretrained=True, freeze_backbone=False, device=device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output.squeeze()}")
