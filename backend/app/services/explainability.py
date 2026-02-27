"""
Explainability Module
- Grad-CAM for fundus model (final convolution block)
- Feature importance mapping for OCT model
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from app.utils.image_processing import overlay_heatmap, numpy_to_base64


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the fundus model.

    Hooks into the last convolutional layer of the EfficientNet backbone
    to capture gradients and activations.

    Args:
        model: QuantumFundusClassifier instance.
        input_tensor: Preprocessed fundus image (1, 3, 224, 224).
        target_class: Class index (1=CSCR, 0=Healthy).

    Returns:
        Heatmap as numpy array of shape (H, W), values in [0, 1].
    """
    model.eval()

    gradients = []
    activations = []

    # Hook into the last conv layer of EfficientNet backbone
    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Get the final block of EfficientNet
    # EfficientNet-B0's last conv feature extraction layer
    target_layer = model.backbone._conv_head

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        output = model(input_tensor)
        prob = torch.sigmoid(output)

        # Backward pass for target class
        model.zero_grad()
        if target_class == 1:
            loss = output.sum()
        else:
            loss = -output.sum()
        loss.backward()

        # Grad-CAM computation
        grads = gradients[0]  # (1, C, H, W)
        acts = activations[0]  # (1, C, H, W)

        # Global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return cam


def compute_oct_feature_importance(
    model: torch.nn.Module,
    features: np.ndarray,
) -> np.ndarray:
    """
    Compute feature importance for the OCT model using gradient-based attribution.

    Computes the gradient of the output with respect to each of the 64 input features
    to determine which features contributed most to the prediction.

    Args:
        model: QuantumOCTClassifier instance.
        features: Array of shape (64,) — extracted OCT features.

    Returns:
        Feature importance array of shape (64,), normalized to [0, 1].
    """
    model.eval()

    # Create tensor with gradient tracking
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    input_tensor.requires_grad_(True)

    # Forward pass
    output = model(input_tensor)

    # Backward pass
    model.zero_grad()
    output.sum().backward()

    # Feature importance = absolute gradient magnitude
    importance = input_tensor.grad.abs().squeeze().cpu().numpy()

    # Normalize
    if importance.max() > 0:
        importance = importance / importance.max()

    return importance


def feature_importance_to_heatmap(
    importance: np.ndarray,
    image_shape: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Convert 64-dim feature importance vector to a spatial heatmap.

    Maps the 64 features back to an 8x8 grid and upscales to image dimensions.

    Args:
        importance: Array of shape (64,).
        image_shape: Target output shape.

    Returns:
        Heatmap array of shape (H, W).
    """
    # Reshape to 8x8 grid
    grid = importance.reshape(8, 8)

    # Upscale to image size
    heatmap = cv2.resize(grid, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def generate_oct_explainability(
    model: torch.nn.Module,
    features: np.ndarray,
    original_image: np.ndarray,
) -> dict:
    """
    Generate full explainability output for OCT prediction.

    Returns:
        Dict with 'importance' (64-vector), 'heatmap_base64' (overlay image).
    """
    importance = compute_oct_feature_importance(model, features)
    heatmap = feature_importance_to_heatmap(importance, original_image.shape[:2])
    overlay = overlay_heatmap(original_image, (heatmap * 255).astype(np.uint8))

    return {
        "feature_importance": importance.tolist(),
        "heatmap_base64": numpy_to_base64(overlay),
    }


def generate_fundus_explainability(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    predicted_class: int,
) -> dict:
    """
    Generate Grad-CAM explainability output for fundus prediction.

    Returns:
        Dict with 'gradcam_base64' (overlay image).
    """
    cam = compute_gradcam(model, input_tensor, target_class=predicted_class)

    # Resize to original image dimensions
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    overlay = overlay_heatmap(original_image, (cam_resized * 255).astype(np.uint8))

    return {
        "gradcam_base64": numpy_to_base64(overlay),
    }
