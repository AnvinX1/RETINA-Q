"""
Pydantic Schemas — Request and Response models for the RETINA-Q API.
"""
from datetime import datetime
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
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None


# ──────────────────────────────────────────────────────────────
# Async Job Schemas
# ──────────────────────────────────────────────────────────────

class JobSubmittedResponse(BaseModel):
    """Returned when an async inference job is dispatched."""
    job_id: str = Field(..., description="Unique job identifier for polling")
    status: str = Field(default="processing", description="Initial job status")


class JobStatusResponse(BaseModel):
    """Polling response for an async inference job."""
    job_id: str
    status: str = Field(..., description="pending | processing | complete | failed")
    step: str | None = Field(None, description="Current processing step (if processing)")
    result: dict | None = Field(None, description="Inference result (if complete)")
    error: str | None = Field(None, description="Error message (if failed)")


# ──────────────────────────────────────────────────────────────
# Doctor Feedback Schemas
# ──────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Doctor feedback submission."""
    job_id: str = Field(..., description="The job_id of the prediction being reviewed")
    doctor_verdict: str = Field(
        ...,
        description="'accept' or 'reject'",
        pattern="^(accept|reject)$",
    )
    correction: str | None = Field(
        None,
        description="Correct diagnosis label if rejecting (e.g. 'Normal', 'CSR', 'CSCR', 'Healthy')",
    )
    notes: str | None = Field(None, description="Optional free-text clinical notes")


class FeedbackResponse(BaseModel):
    """Response after recording feedback."""
    job_id: str
    status: str = Field(default="recorded")
    message: str


# ──────────────────────────────────────────────────────────────
# Patient Schemas
# ──────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    """Create a new patient."""
    patient_id: str = Field(..., min_length=1, max_length=64, description="Clinic-assigned patient ID")
    name: str = Field(..., min_length=1, max_length=255, description="Patient full name")
    age: int | None = Field(None, ge=0, le=150, description="Patient age")
    gender: str | None = Field(None, max_length=16, description="Patient gender")
    medical_history: str | None = Field(None, description="Relevant medical history")
    notes: str | None = Field(None, description="Additional clinical notes")


class PatientUpdate(BaseModel):
    """Update patient fields (all optional)."""
    name: str | None = Field(None, min_length=1, max_length=255)
    age: int | None = Field(None, ge=0, le=150)
    gender: str | None = Field(None, max_length=16)
    medical_history: str | None = None
    notes: str | None = None


class ScanResponse(BaseModel):
    """A single scan record."""
    id: int
    job_id: str
    patient_id: int | None = None
    image_type: str
    prediction: str
    confidence: float
    probability: float
    feedback_status: str = "pending"
    doctor_notes: str | None = None
    mask_area_ratio: float | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PatientResponse(BaseModel):
    """Patient detail with scan history."""
    id: int
    patient_id: str
    name: str
    age: int | None = None
    gender: str | None = None
    medical_history: str | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime
    scans: list[ScanResponse] = []

    model_config = {"from_attributes": True}


class PatientListItem(BaseModel):
    """Patient summary for list views (no scans)."""
    id: int
    patient_id: str
    name: str
    age: int | None = None
    gender: str | None = None
    scan_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    """Generic paginated wrapper."""
    items: list = []
    total: int = 0
    page: int = 1
    per_page: int = 20
