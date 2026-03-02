"""
Scan ORM Model — stores each diagnostic scan result linked to a patient.
"""
from datetime import datetime, timezone
from sqlalchemy import Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    patient_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("patients.id", ondelete="SET NULL"), nullable=True
    )
    image_type: Mapped[str] = mapped_column(String(16), nullable=False)  # "oct" or "fundus"
    prediction: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    # Stored image paths (relative to uploads/scans/)
    original_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    gradcam_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    segmentation_mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    segmentation_overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mask_area_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Feature importance stored as JSON text
    feature_importance_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Doctor feedback
    feedback_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/accepted/rejected
    doctor_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    patient = relationship("Patient", back_populates="scans")

    def __repr__(self) -> str:
        return f"<Scan {self.job_id}: {self.image_type} → {self.prediction}>"
