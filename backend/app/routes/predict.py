"""
Prediction Routes — OCT and Fundus classification endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.schemas.responses import (
    OCTPredictionResponse,
    FundusPredictionResponse,
    ErrorResponse,
)
from app.services.inference import run_oct_inference, run_fundus_inference
from app.config import settings


router = APIRouter(prefix="/api/predict", tags=["Prediction"])


@router.post(
    "/oct",
    response_model=OCTPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify OCT Image",
    description="Upload a grayscale OCT image for quantum-enhanced binary classification (Normal vs CSR).",
)
async def predict_oct(file: UploadFile = File(..., description="OCT image file (JPEG/PNG)")):
    """OCT classification endpoint using the 8-qubit quantum circuit."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    try:
        image_bytes = await file.read()

        # Check file size
        if len(image_bytes) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.max_upload_size_mb}MB limit",
            )

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
    response_model=FundusPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify Fundus Image",
    description="Upload a fundus image for quantum-enhanced classification (Healthy vs CSCR) with optional segmentation.",
)
async def predict_fundus(
    file: UploadFile = File(..., description="Fundus image file (JPEG/PNG)"),
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

        logger.info(f"Fundus prediction request — file: {file.filename}, size: {len(image_bytes)} bytes")
        result = run_fundus_inference(image_bytes, run_segmentation=True)
        logger.info(f"Fundus prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")

        return FundusPredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fundus prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
