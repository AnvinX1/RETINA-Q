"""
RETINA-Q Configuration
"""
import os
from pathlib import Path
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "weights"
FEEDBACK_DIR = BASE_DIR / "feedback"
QUARANTINE_DIR = FEEDBACK_DIR / "quarantine"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    app_name: str = "RETINA-Q"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Model settings
    oct_model_path: str = str(MODEL_DIR / "oct_quantum.pth")
    fundus_model_path: str = str(MODEL_DIR / "fundus_quantum.pth")
    unet_model_path: str = str(MODEL_DIR / "unet_segmentation.pth")

    # Quantum settings
    oct_num_qubits: int = 8
    fundus_num_qubits: int = 4
    quantum_device: str = "default.qubit"

    # Image settings
    oct_image_size: tuple[int, int] = (224, 224)
    fundus_image_size: tuple[int, int] = (224, 224)
    segmentation_image_size: tuple[int, int] = (256, 256)

    # Inference
    confidence_threshold: float = 0.5
    max_upload_size_mb: int = 10

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "capacitor://localhost",
        "http://localhost",
    ]

    # ── Celery / Redis ──────────────────────────────────────
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # ── MLflow ──────────────────────────────────────────────
    mlflow_enabled: bool = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # ── Feedback ────────────────────────────────────────────
    feedback_log_path: str = str(FEEDBACK_DIR / "feedback_log.jsonl")
    shadow_log_path: str = str(FEEDBACK_DIR / "shadow_log.jsonl")
    quarantine_dir: str = str(QUARANTINE_DIR)

    # ── Shadow Deployment ───────────────────────────────────
    shadow_enabled: bool = os.getenv("SHADOW_ENABLED", "false").lower() == "true"


settings = Settings()
