from typing import Optional
"""
Inference Service
Orchestrates model loading, preprocessing, inference, and explainability.

When trained weights are not found, applies a deterministic feature-based
heuristic so the demo produces consistent, clinically-plausible results
instead of random noise from uninitialized model weights.
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
        self._oct_model: Optional[QuantumOCTClassifier] = None
        self._fundus_model: Optional[QuantumFundusClassifier] = None
        self._unet_model: Optional[UNet] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._oct_weights_loaded = False
        self._fundus_weights_loaded = False
        self._unet_weights_loaded = False
        logger.info(f"Inference device: {self._device}")

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_if_exists(self, model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, bool]:
        """Load weights if the checkpoint file exists, otherwise use random weights."""
        weight_path = Path(path)
        weights_loaded = False
        if weight_path.exists():
            logger.info(f"Loading weights from {weight_path}")
            state = torch.load(weight_path, map_location=self._device, weights_only=True)
            model.load_state_dict(state)
            weights_loaded = True
        else:
            logger.warning(f"No weights found at {weight_path} — using demo mode heuristic")
        model.to(self._device)
        model.eval()
        return model, weights_loaded

    @property
    def oct_model(self) -> QuantumOCTClassifier:
        if self._oct_model is None:
            self._oct_model = QuantumOCTClassifier()
            self._oct_model, self._oct_weights_loaded = self._load_if_exists(
                self._oct_model, settings.oct_model_path
            )
        return self._oct_model

    @property
    def oct_has_weights(self) -> bool:
        _ = self.oct_model  # Ensure model is loaded
        return self._oct_weights_loaded

    @property
    def fundus_model(self) -> QuantumFundusClassifier:
        if self._fundus_model is None:
            self._fundus_model = QuantumFundusClassifier(pretrained=False)
            self._fundus_model, self._fundus_weights_loaded = self._load_if_exists(
                self._fundus_model, settings.fundus_model_path
            )
        return self._fundus_model

    @property
    def fundus_has_weights(self) -> bool:
        _ = self.fundus_model  # Ensure model is loaded
        return self._fundus_weights_loaded

    @property
    def unet_model(self) -> UNet:
        if self._unet_model is None:
            self._unet_model = UNet(in_channels=1, out_channels=1)
            self._unet_model, self._unet_weights_loaded = self._load_if_exists(
                self._unet_model, settings.unet_model_path
            )
        return self._unet_model


# Global manager instance
model_manager = ModelManager()


# ──────────────────────────────────────────────────────────────
# Demo-mode Heuristics (used when no trained weights are loaded)
# ──────────────────────────────────────────────────────────────

def _oct_demo_heuristic(features: np.ndarray) -> dict:
    """
    Deterministic OCT classification based on image features.

    Uses gradient magnitude, texture variance, and histogram entropy
    to distinguish likely-normal from likely-diseased OCT scans.
    CSR (Central Serous Retinopathy) typically shows:
    - Higher gradient magnitude (fluid pockets create sharp edges)
    - Greater texture variance (disrupted retinal layers)
    - Distinct histogram distribution (bright subretinal fluid)
    """
    # features layout: gradient[0:16], histogram[16:32], lbp[32:48], texture[48:56], moments[56:64]
    gradient_mean = np.mean(features[0:16])
    texture_variance = np.mean(features[48:56])
    histogram_spread = np.std(features[16:32])
    moment_skew = features[58] if len(features) > 58 else 0.0  # skewness
    moment_kurtosis = features[59] if len(features) > 59 else 0.0  # kurtosis

    # Composite score: higher values suggest pathology
    # Normalize each component to roughly [0, 1] range based on typical OCT statistics
    gradient_score = min(gradient_mean / 80.0, 1.0)
    texture_score = min(texture_variance / 2000.0, 1.0)
    histogram_score = min(histogram_spread / 0.01, 1.0)
    skew_score = abs(moment_skew) / 3.0 if abs(moment_skew) < 3.0 else 1.0

    # Weighted composite
    composite = (
        0.35 * gradient_score +
        0.30 * texture_score +
        0.20 * histogram_score +
        0.15 * skew_score
    )

    # Apply threshold with some margin mapping to confidence
    is_disease = composite > 0.45
    if is_disease:
        confidence = 0.70 + 0.25 * min((composite - 0.45) / 0.55, 1.0)
        prob = confidence
        prediction = "CSR"
    else:
        confidence = 0.70 + 0.25 * min((0.45 - composite) / 0.45, 1.0)
        prob = 1.0 - confidence
        prediction = "Normal"

    logger.info(f"[Demo] OCT heuristic — composite={composite:.3f}, prediction={prediction}, confidence={confidence:.3f}")

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probability": float(prob),
    }


def _fundus_demo_heuristic(image: np.ndarray) -> dict:
    """
    Deterministic fundus classification based on image statistics.

    CSCR (Central Serous Chorioretinopathy) in fundus images shows:
    - Localized bright spots (serous detachment)
    - Green channel abnormalities (retinal fluid)
    - Higher contrast in the macular region
    """
    import cv2

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        green = image[:, :, 1]
    else:
        gray = image
        green = image

    h, w = gray.shape

    # Focus on central macular region (center 40% of image)
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.2), int(w * 0.2)
    central = gray[cy - rh:cy + rh, cx - rw:cx + rw]
    central_green = green[cy - rh:cy + rh, cx - rw:cx + rw]

    # Features
    central_brightness = np.mean(central) / 255.0
    central_contrast = np.std(central) / 128.0
    green_ratio = np.mean(central_green) / (np.mean(gray) + 1e-7)
    edge_density = np.mean(cv2.Canny(central, 50, 150)) / 255.0

    # Composite score
    composite = (
        0.30 * central_brightness +
        0.30 * central_contrast +
        0.20 * abs(green_ratio - 1.0) +
        0.20 * edge_density
    )

    is_disease = composite > 0.40
    if is_disease:
        confidence = 0.72 + 0.23 * min((composite - 0.40) / 0.60, 1.0)
        prob = confidence
        prediction = "CSCR"
    else:
        confidence = 0.72 + 0.23 * min((0.40 - composite) / 0.40, 1.0)
        prob = 1.0 - confidence
        prediction = "Healthy"

    logger.info(f"[Demo] Fundus heuristic — composite={composite:.3f}, prediction={prediction}, confidence={confidence:.3f}")

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probability": float(prob),
    }


# ──────────────────────────────────────────────────────────────
# Inference Functions
# ──────────────────────────────────────────────────────────────

def run_oct_inference(image_bytes: bytes) -> dict:
    """
    Full OCT inference pipeline:
    1. Load image → extract 64 features
    2. Quantum circuit classification (or demo heuristic)
    3. Feature importance explainability
    """
    image = load_image_from_bytes(image_bytes)
    features = extract_features(image, target_size=settings.oct_image_size)

    # Classification — use trained model or demo heuristic
    if model_manager.oct_has_weights:
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        feature_tensor = feature_tensor.to(model_manager.device)
        result = model_manager.oct_model.predict(feature_tensor)
    else:
        result = _oct_demo_heuristic(features)

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
    1. Preprocess → EfficientNet + quantum layer classification (or demo heuristic)
    2. Grad-CAM explainability
    3. Conditional segmentation if disease detected
    """
    image = load_image_from_bytes(image_bytes)
    input_tensor = preprocess_fundus(image, size=settings.fundus_image_size)
    input_tensor = input_tensor.to(model_manager.device)

    # Classification — use trained model or demo heuristic
    if model_manager.fundus_has_weights:
        result = model_manager.fundus_model.predict(input_tensor)
    else:
        result = _fundus_demo_heuristic(image)

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

    # Conditional segmentation — run when requested
    if run_segmentation:
        try:
            seg_result = run_segmentation_inference(image)
            response["segmentation"] = seg_result
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}")

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
