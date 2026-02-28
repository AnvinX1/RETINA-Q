"""
Feedback Routes — Doctor accept/reject endpoint for clinical feedback loop.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.schemas.responses import FeedbackRequest, FeedbackResponse
from app.config import settings


router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Submit Doctor Feedback",
    description="Allow clinicians to accept or reject an AI prediction, building a feedback dataset for retraining.",
)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Records doctor feedback for a given job.
    - If 'reject', the original image is copied to the quarantine directory
      tagged with the doctor's correction label for future retraining.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": feedback.job_id,
        "doctor_verdict": feedback.doctor_verdict,
        "correction": feedback.correction,
        "notes": feedback.notes,
    }

    # ── Append to feedback log ──────────────────────────────
    try:
        with open(settings.feedback_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Feedback recorded — job={feedback.job_id}, verdict={feedback.doctor_verdict}")
    except Exception as e:
        logger.error(f"Failed to write feedback log: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    # ── Quarantine rejected images ──────────────────────────
    if feedback.doctor_verdict == "reject" and feedback.correction:
        try:
            # Look for the original upload by job_id
            upload_dir = Path(settings.quarantine_dir).parent.parent / "uploads"
            matching = list(upload_dir.glob(f"{feedback.job_id}.*"))
            if matching:
                src = matching[0]
                label_dir = Path(settings.quarantine_dir) / feedback.correction
                label_dir.mkdir(parents=True, exist_ok=True)
                dst = label_dir / src.name
                shutil.copy2(str(src), str(dst))
                logger.info(f"Quarantined image: {src.name} → {feedback.correction}/")
        except Exception as e:
            logger.warning(f"Failed to quarantine image: {e}")

    return FeedbackResponse(
        job_id=feedback.job_id,
        status="recorded",
        message=f"Feedback '{feedback.doctor_verdict}' recorded successfully.",
    )
