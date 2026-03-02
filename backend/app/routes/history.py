"""
Scan History Routes — browse and manage diagnostic scan records.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.db_models.scan import Scan
from app.schemas.responses import ScanResponse, PaginatedResponse


router = APIRouter(prefix="/api/scans", tags=["Scan History"])


@router.get("", response_model=PaginatedResponse)
def list_scans(
    patient_id: int | None = Query(None, description="Filter by patient DB id"),
    image_type: str | None = Query(None, description="Filter by image type: oct or fundus"),
    prediction: str | None = Query(None, description="Filter by prediction label"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List scans with optional filters and pagination."""
    query = db.query(Scan)

    if patient_id is not None:
        query = query.filter(Scan.patient_id == patient_id)
    if image_type:
        query = query.filter(Scan.image_type == image_type)
    if prediction:
        query = query.filter(Scan.prediction == prediction)

    total = query.count()
    scans = query.order_by(Scan.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    items = [ScanResponse.model_validate(s).model_dump() for s in scans]
    return PaginatedResponse(items=items, total=total, page=page, per_page=per_page)


@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    """Get a single scan record."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@router.delete("/{scan_id}", status_code=204)
def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    """Delete a scan record."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    db.delete(scan)
    db.commit()
    logger.info(f"Scan deleted — id={scan_id}, job_id={scan.job_id}")
