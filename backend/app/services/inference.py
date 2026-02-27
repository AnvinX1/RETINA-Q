"""
Inference Service
Orchestrates model loading, preprocessing, inference, and explainability.
"""
import torch
import numpy as np
from loguru import logger
from pathlib import Path

from app.config import settings
from app.models.quantum_oct_model import QuantumOCTClassifier
from app.models.quantum_fundus_model import QuantumFundusClassifier
from app.models.unet_model import UNet
from app.utils.oct_feature_extractor import extract_features
from app.utils.image_processing import (
    load_image_from_bytes,
    preprocess_fundus,
    preprocess_segmentation,
    postprocess_mask,
    numpy_to_base64,
    overlay_heatmap,
)
from app.services.explainability import (
    generate_oct_explainability,
    generate_fundus_explainability,
)


class ModelManager:
    """Singleton-style model manager. Lazy-loads models on first use."""

    def __init__(self):
        self._oct_model: QuantumOCTClassifier | None = None
        self._fundus_model: QuantumFundusClassifier | None = None
        self._unet_model: UNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self._device}")

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_if_exists(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        """Load weights if the checkpoint file exists, otherwise use random weights."""
        weight_path = Path(path)
        if weight_path.exists():
            logger.info(f"Loading weights from {weight_path}")
            state = torch.load(weight_path, map_location=self._device, weights_only=True)
            model.load_state_dict(state)
        else:
            logger.warning(f"No weights found at {weight_path} — using random initialisation")
        model.to(self._device)
        model.eval()
        return model

    @property
    def oct_model(self) -> QuantumOCTClassifier:
        if self._oct_model is None:
            self._oct_model = QuantumOCTClassifier()
            self._oct_model = self._load_if_exists(self._oct_model, settings.oct_model_path)
        return self._oct_model

    @property
    def fundus_model(self) -> QuantumFundusClassifier:
        if self._fundus_model is None:
            self._fundus_model = QuantumFundusClassifier(pretrained=False)
            self._fundus_model = self._load_if_exists(self._fundus_model, settings.fundus_model_path)
        return self._fundus_model

    @property
    def unet_model(self) -> UNet:
        if self._unet_model is None:
            self._unet_model = UNet(in_channels=1, out_channels=1)
            self._unet_model = self._load_if_exists(self._unet_model, settings.unet_model_path)
        return self._unet_model


# Global manager instance
model_manager = ModelManager()


# ──────────────────────────────────────────────────────────────
# Inference Functions
# ──────────────────────────────────────────────────────────────

def run_oct_inference(image_bytes: bytes) -> dict:
    """
    Full OCT inference pipeline:
    1. Load image → extract 64 features
    2. Quantum circuit classification
    3. Feature importance explainability
    """
    image = load_image_from_bytes(image_bytes)
    features = extract_features(image, target_size=settings.oct_image_size)

    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    feature_tensor = feature_tensor.to(model_manager.device)

    result = model_manager.oct_model.predict(feature_tensor)

    # Explainability
    explain = generate_oct_explainability(
        model_manager.oct_model, features, image,
    )

    return {
        **result,
        "image_type": "OCT",
        "heatmap_base64": explain["heatmap_base64"],
        "feature_importance": explain["feature_importance"],
    }


def run_fundus_inference(image_bytes: bytes, run_segmentation: bool = True) -> dict:
    """
    Full Fundus inference pipeline:
    1. Preprocess → EfficientNet + quantum layer classification
    2. Grad-CAM explainability
    3. Conditional segmentation if disease detected
    """
    image = load_image_from_bytes(image_bytes)
    input_tensor = preprocess_fundus(image, size=settings.fundus_image_size)
    input_tensor = input_tensor.to(model_manager.device)

    result = model_manager.fundus_model.predict(input_tensor)

    # Grad-CAM
    predicted_class = 1 if result["prediction"] == "CSCR" else 0
    explain = generate_fundus_explainability(
        model_manager.fundus_model, input_tensor, image, predicted_class,
    )

    response = {
        **result,
        "image_type": "Fundus",
        "gradcam_base64": explain["gradcam_base64"],
        "segmentation": None,
    }

    # Conditional segmentation — only if disease detected and requested
    if run_segmentation and result["prediction"] == "CSCR":
        seg_result = run_segmentation_inference(image)
        response["segmentation"] = seg_result

    return response


def run_segmentation_inference(image_or_bytes) -> dict:
    """
    Macular segmentation pipeline:
    1. Green channel extraction + CLAHE
    2. U-Net inference
    3. Morphological post-processing
    """
    if isinstance(image_or_bytes, bytes):
        image = load_image_from_bytes(image_or_bytes)
    else:
        image = image_or_bytes

    original_size = image.shape[:2]

    seg_tensor = preprocess_segmentation(image, size=settings.segmentation_image_size)
    seg_tensor = seg_tensor.to(model_manager.device)

    with torch.no_grad():
        mask_pred = model_manager.unet_model(seg_tensor)

    mask_np = mask_pred.squeeze().cpu().numpy()
    refined_mask = postprocess_mask(mask_np, original_size)

    # Create overlay
    overlay = image.copy()
    colored_mask = np.zeros_like(overlay)
    colored_mask[:, :, 1] = refined_mask  # Green channel for mask overlay
    overlay = (0.7 * overlay + 0.3 * colored_mask).astype(np.uint8)

    return {
        "mask_base64": numpy_to_base64(refined_mask),
        "overlay_base64": numpy_to_base64(overlay),
        "mask_area_ratio": float(refined_mask.sum() / (refined_mask.size * 255)),
    }
