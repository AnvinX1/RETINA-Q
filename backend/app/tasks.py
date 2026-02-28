"""
RETINA-Q — Celery Tasks

Background tasks that perform heavy ML inference off the HTTP thread.
Each task reads a saved image from disk, runs the model, and returns
the result dict which Celery stores in the Redis result backend.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

from app.celery_app import celery_app
from app.config import settings
from app.services.inference import (
    run_oct_inference,
    run_fundus_inference,
    run_segmentation_inference,
)


def _log_shadow(job_id: str, image_type: str, shadow_result: dict) -> None:
    """Append a shadow model prediction to the shadow log (JSONL)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "image_type": image_type,
        "shadow_prediction": shadow_result.get("prediction"),
        "shadow_confidence": shadow_result.get("confidence"),
    }
    with open(settings.shadow_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


@celery_app.task(bind=True, name="retinaq.predict_oct")
def predict_oct_task(self, job_id: str, filepath: str):
    """Run OCT inference in a background worker."""
    logger.info(f"[Task] OCT inference started — job={job_id}")
    self.update_state(state="PROCESSING", meta={"step": "loading_image"})

    image_bytes = Path(filepath).read_bytes()

    self.update_state(state="PROCESSING", meta={"step": "quantum_inference"})
    result = run_oct_inference(image_bytes)

    # ── Shadow deployment ───────────────────────────────────
    if settings.shadow_enabled:
        try:
            shadow_result = run_oct_inference(image_bytes)  # Would use shadow model
            _log_shadow(job_id, "oct", shadow_result)
        except Exception as e:
            logger.warning(f"Shadow OCT inference failed: {e}")

    logger.info(f"[Task] OCT inference complete — job={job_id}, pred={result['prediction']}")
    return {"job_id": job_id, "image_type": "oct", **result}


@celery_app.task(bind=True, name="retinaq.predict_fundus")
def predict_fundus_task(self, job_id: str, filepath: str):
    """Run Fundus inference in a background worker."""
    logger.info(f"[Task] Fundus inference started — job={job_id}")
    self.update_state(state="PROCESSING", meta={"step": "loading_image"})

    image_bytes = Path(filepath).read_bytes()

    self.update_state(state="PROCESSING", meta={"step": "quantum_inference"})
    result = run_fundus_inference(image_bytes, run_segmentation=True)

    # ── Shadow deployment ───────────────────────────────────
    if settings.shadow_enabled:
        try:
            shadow_result = run_fundus_inference(image_bytes, run_segmentation=False)
            _log_shadow(job_id, "fundus", shadow_result)
        except Exception as e:
            logger.warning(f"Shadow Fundus inference failed: {e}")

    logger.info(f"[Task] Fundus inference complete — job={job_id}, pred={result['prediction']}")
    return {"job_id": job_id, "image_type": "fundus", **result}


@celery_app.task(bind=True, name="retinaq.segment")
def segment_task(self, job_id: str, filepath: str):
    """Run standalone segmentation in a background worker."""
    logger.info(f"[Task] Segmentation started — job={job_id}")
    self.update_state(state="PROCESSING", meta={"step": "loading_image"})

    image_bytes = Path(filepath).read_bytes()

    self.update_state(state="PROCESSING", meta={"step": "unet_inference"})
    result = run_segmentation_inference(image_bytes)

    logger.info(f"[Task] Segmentation complete — job={job_id}")
    return {"job_id": job_id, "image_type": "segmentation", **result}
