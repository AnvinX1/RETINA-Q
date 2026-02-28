"""
Job Status Routes — Polling and SSE endpoints for async inference results.
"""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.celery_app import celery_app
from app.schemas.responses import JobStatusResponse


router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get Job Status",
    description="Poll the status and result of an async inference job.",
)
async def get_job_status(job_id: str):
    """Check the status of a Celery task by job_id."""
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return JobStatusResponse(job_id=job_id, status="pending")
    elif result.state == "PROCESSING":
        meta = result.info or {}
        return JobStatusResponse(
            job_id=job_id,
            status="processing",
            step=meta.get("step", "unknown"),
        )
    elif result.state == "SUCCESS":
        return JobStatusResponse(
            job_id=job_id,
            status="complete",
            result=result.result,
        )
    elif result.state == "FAILURE":
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(result.info),
        )
    else:
        return JobStatusResponse(job_id=job_id, status=result.state.lower())


@router.get(
    "/{job_id}/stream",
    summary="Stream Job Status (SSE)",
    description="Server-Sent Events stream for real-time job status updates.",
)
async def stream_job_status(job_id: str):
    """SSE endpoint — pushes status updates until the job completes or fails."""

    async def event_generator():
        while True:
            result = celery_app.AsyncResult(job_id)
            state = result.state

            if state == "PENDING":
                yield f"data: {json.dumps({'status': 'pending'})}\n\n"
            elif state == "PROCESSING":
                meta = result.info or {}
                yield f"data: {json.dumps({'status': 'processing', 'step': meta.get('step', 'unknown')})}\n\n"
            elif state == "SUCCESS":
                yield f"data: {json.dumps({'status': 'complete', 'result': result.result})}\n\n"
                return  # Close the stream
            elif state == "FAILURE":
                yield f"data: {json.dumps({'status': 'failed', 'error': str(result.info)})}\n\n"
                return  # Close the stream
            else:
                yield f"data: {json.dumps({'status': state.lower()})}\n\n"

            await asyncio.sleep(1)  # Poll every second

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
