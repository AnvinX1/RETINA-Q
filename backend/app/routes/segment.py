"""
Segmentation Route — standalone macular segmentation endpoint.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.schemas.responses import SegmentationResult, ErrorResponse
from app.services.inference import run_segmentation_inference
from app.config import settings


router = APIRouter(prefix="/api", tags=["Segmentation"])


@router.post(
    "/segment",
    response_model=SegmentationResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Segment Fundus Image",
    description="Upload a fundus image for U-Net macular segmentation with morphological post-processing.",
)
async def segment(file: UploadFile = File(..., description="Fundus image file (JPEG/PNG)")):
    """Standalone segmentation endpoint using U-Net."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    try:
        image_bytes = await file.read()

        if len(image_bytes) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.max_upload_size_mb}MB limit",
            )

        logger.info(f"Segmentation request — file: {file.filename}, size: {len(image_bytes)} bytes")
        result = run_segmentation_inference(image_bytes)
        logger.info(f"Segmentation complete — mask area ratio: {result['mask_area_ratio']:.4f}")

        return SegmentationResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
