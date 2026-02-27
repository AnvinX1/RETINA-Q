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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]


settings = Settings()
