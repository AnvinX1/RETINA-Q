"""
Pydantic Schemas — Request and Response models for the RETINA-Q API.
"""
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────────────────────

class SegmentationResult(BaseModel):
    """Segmentation output."""
    mask_base64: str = Field(..., description="Base64-encoded PNG of the binary segmentation mask")
    overlay_base64: str = Field(..., description="Base64-encoded PNG of mask overlaid on original image")
    mask_area_ratio: float = Field(..., description="Ratio of segmented area to total image area")


class OCTPredictionResponse(BaseModel):
    """Response for OCT classification."""
    image_type: str = Field(default="OCT")
    prediction: str = Field(..., description="Classification result: 'Normal' or 'CSR'")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probability: float = Field(..., ge=0, le=1, description="Raw sigmoid probability")
    heatmap_base64: str = Field(..., description="Base64-encoded feature importance heatmap overlay")
    feature_importance: list[float] = Field(..., description="64 feature importance values")


class FundusPredictionResponse(BaseModel):
    """Response for Fundus classification with optional segmentation."""
    image_type: str = Field(default="Fundus")
    prediction: str = Field(..., description="Classification result: 'Healthy' or 'CSCR'")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probability: float = Field(..., ge=0, le=1, description="Raw sigmoid probability")
    gradcam_base64: str = Field(..., description="Base64-encoded Grad-CAM heatmap overlay")
    segmentation: SegmentationResult | None = Field(
        None, description="Segmentation result (only when CSCR detected)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str = "RETINA-Q"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None
