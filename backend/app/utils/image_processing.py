"""
Image Processing Utilities
Common image loading, preprocessing, and transformation functions.
"""
import io
import base64

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load an image from raw bytes into a numpy array (BGR)."""
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from provided bytes")
    return image


def preprocess_fundus(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Preprocess a fundus image for EfficientNet inference."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transform(pil_image).unsqueeze(0)  # (1, 3, H, W)


def preprocess_segmentation(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Preprocess a fundus image for U-Net segmentation.
    - Extract green channel
    - Apply CLAHE enhancement
    """
    # Extract green channel (best contrast for retinal structures)
    if len(image.shape) == 3:
        green = image[:, :, 1]
    else:
        green = image

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Resize and normalize
    resized = cv2.resize(enhanced, size)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, H, W)

    return tensor


def postprocess_mask(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """
    Post-process segmentation mask:
    - Binarize
    - Connected components → keep largest
    - Morphological refinement
    """
    # Binarize
    binary = (mask > 0.5).astype(np.uint8) * 255

    # Connected components — keep largest
    num_labels, labels, region_stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Ignore background (label 0)
        largest_label = 1 + np.argmax(region_stats[1:, cv2.CC_STAT_AREA])
        binary = ((labels == largest_label) * 255).astype(np.uint8)

    # Morphological refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Resize to original
    binary = cv2.resize(binary, (original_size[1], original_size[0]))

    return binary


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert a numpy image to a base64-encoded PNG string."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a heatmap on an image with transparency."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize to 0-255
    if heatmap_resized.max() > 0:
        heatmap_resized = (heatmap_resized / heatmap_resized.max() * 255).astype(np.uint8)

    colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

    return overlay
