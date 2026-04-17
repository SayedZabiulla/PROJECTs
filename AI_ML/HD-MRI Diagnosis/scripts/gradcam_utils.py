# scripts/gradcam_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """Grad-CAM implementation for visualizing model attention"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save activations during forward pass"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for CAM (None = predicted class)
        
        Returns:
            heatmap: Grad-CAM heatmap (H, W)
            prediction: Predicted class
            prob: Prediction probability
        """
        # Ensure input is on the same device as model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get prediction
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob >= 0.5 else 0
        
        # Use target class or predicted class
        if target_class is None:
            target_class = prediction
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        if target_class == 1:
            output.backward()
        else:
            (-output).backward()
        
        # Get activations and gradients
        activations = self.activations  # (1, C, H, W) - on CUDA
        gradients = self.gradients      # (1, C, H, W) - on CUDA
        
        # Global average pooling on gradients to get weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1) - on CUDA
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W) - on CUDA
        
        # Apply ReLU and normalize
        cam = F.relu(cam)  # Only keep positive influences
        cam = cam.squeeze().cpu().numpy()  # Move to CPU for numpy operations
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
        
        return cam, prediction, prob
    
    def __del__(self):
        """Remove hooks when object is deleted"""
        self.forward_hook.remove()
        self.backward_hook.remove()


def visualize_gradcam(original_image, heatmap, prediction, true_label, prob, save_path):
    """
    Create and save Grad-CAM visualization
    
    Args:
        original_image: Original MRI slice (H, W)
        heatmap: Grad-CAM heatmap (H, W)
        prediction: Predicted class
        true_label: True class label
        prob: Prediction probability
        save_path: Path to save visualization
    """
    # Convert to numpy if tensor
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    
    # Ensure 2D
    if original_image.ndim == 3:
        original_image = original_image.squeeze()
    
    # Normalize original image to [0, 1]
    img_normalized = original_image.copy()
    if img_normalized.max() > 0:
        img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img_normalized.shape[1], img_normalized.shape[0]))
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Convert grayscale to RGB for overlay
    img_rgb = np.stack([img_normalized] * 3, axis=-1)
    
    # Create overlay
    overlay = 0.5 * img_rgb + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_normalized, cmap='gray')
    axes[0].set_title('Original MRI Slice', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add prediction info
    class_names = ['Control', 'Dementia']
    status = '✓' if prediction == true_label else '✗'
    color = 'green' if prediction == true_label else 'red'
    
    info_text = f"{status} True: {class_names[true_label]} | Pred: {class_names[prediction]} ({prob:.1%})"
    fig.suptitle(info_text, fontsize=14, fontweight='bold', color=color)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def apply_gradcam_batch(model, target_layer, dataloader, device, num_samples=10):
    """
    Apply Grad-CAM to a batch of samples
    
    Args:
        model: Trained model
        target_layer: Layer to visualize
        dataloader: DataLoader with samples
        device: Device to use
        num_samples: Number of samples to visualize
    
    Returns:
        results: List of (image, heatmap, prediction, true_label, prob) tuples
    """
    grad_cam = GradCAM(model, target_layer)
    results = []
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if len(results) >= num_samples:
                break
            
            for i in range(images.size(0)):
                if len(results) >= num_samples:
                    break
                
                # Get single image
                img = images[i:i+1].to(device)
                label = labels[i].item()
                
                # Generate CAM (requires gradients)
                heatmap, prediction, prob = grad_cam.generate_cam(img)
                
                # Store results
                original_img = images[i].cpu().numpy().squeeze()
                results.append({
                    'image': original_img,
                    'heatmap': heatmap,
                    'prediction': prediction,
                    'true_label': label,
                    'prob': prob
                })
    
    return results
