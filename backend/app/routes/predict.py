"""
Prediction Routes — OCT and Fundus classification endpoints.

Now supports both synchronous (direct) and asynchronous (Celery) modes.
When Celery is available, endpoints return 202 Accepted with a job_id.
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from loguru import logger

from app.schemas.responses import (
    OCTPredictionResponse,
    FundusPredictionResponse,
    JobSubmittedResponse,
    ErrorResponse,
)
from app.services.inference import run_oct_inference, run_fundus_inference
from app.config import settings, UPLOAD_DIR


router = APIRouter(prefix="/api/predict", tags=["Prediction"])


def _save_upload(image_bytes: bytes, filename: str) -> tuple[str, str]:
    """Save uploaded bytes to disk, return (job_id, filepath)."""
    job_id = str(uuid.uuid4())
    ext = Path(filename or "image.jpg").suffix or ".jpg"
    filepath = UPLOAD_DIR / f"{job_id}{ext}"
    filepath.write_bytes(image_bytes)
    return job_id, str(filepath)


@router.post(
    "/oct",
    responses={
        200: {"model": OCTPredictionResponse},
        202: {"model": JobSubmittedResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Classify OCT Image",
    description="Upload a grayscale OCT image for quantum-enhanced binary classification (Normal vs CSR).",
)
async def predict_oct(
    file: UploadFile = File(..., description="OCT image file (JPEG/PNG)"),
    async_mode: bool = Query(True, description="If true, offload to Celery worker and return job_id"),
):
    """OCT classification endpoint using the 8-qubit quantum circuit."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    try:
        image_bytes = await file.read()

        if len(image_bytes) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.max_upload_size_mb}MB limit",
            )

        # ── Async mode: dispatch to Celery ──────────────────
        if async_mode:
            from app.tasks import predict_oct_task

            job_id, filepath = _save_upload(image_bytes, file.filename)
            predict_oct_task.delay(job_id, filepath)
            logger.info(f"OCT job dispatched — job_id={job_id}")
            return JobSubmittedResponse(job_id=job_id)

        # ── Sync mode: run inline (legacy/testing) ──────────
        logger.info(f"OCT prediction request — file: {file.filename}, size: {len(image_bytes)} bytes")
        result = run_oct_inference(image_bytes)
        logger.info(f"OCT prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return OCTPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCT prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/fundus",
    responses={
        200: {"model": FundusPredictionResponse},
        202: {"model": JobSubmittedResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Classify Fundus Image",
    description="Upload a fundus image for quantum-enhanced classification (Healthy vs CSCR) with optional segmentation.",
)
async def predict_fundus(
    file: UploadFile = File(..., description="Fundus image file (JPEG/PNG)"),
    async_mode: bool = Query(True, description="If true, offload to Celery worker and return job_id"),
):
    """Fundus classification endpoint with conditional macular segmentation."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    try:
        image_bytes = await file.read()

        if len(image_bytes) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.max_upload_size_mb}MB limit",
            )

        # ── Async mode: dispatch to Celery ──────────────────
        if async_mode:
            from app.tasks import predict_fundus_task

            job_id, filepath = _save_upload(image_bytes, file.filename)
            predict_fundus_task.delay(job_id, filepath)
            logger.info(f"Fundus job dispatched — job_id={job_id}")
            return JobSubmittedResponse(job_id=job_id)

        # ── Sync mode: run inline (legacy/testing) ──────────
        logger.info(f"Fundus prediction request — file: {file.filename}, size: {len(image_bytes)} bytes")
        result = run_fundus_inference(image_bytes, run_segmentation=True)
        logger.info(f"Fundus prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return FundusPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fundus prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
